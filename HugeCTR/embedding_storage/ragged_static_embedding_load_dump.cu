/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <curand_kernel.h>

#include <data_simulator.hpp>
#include <embedding/operators/generic_lookup.cuh>
#include <embedding/view.hpp>
#include <embedding_storage/ragged_static_embedding.hpp>
#include <numeric>
#include <utils.cuh>

namespace embedding {

template <typename key_t, typename index_t, typename emb_t>
__global__ void embedding_insert_kernel(
    const key_t *keys, size_t num_keys, const uint32_t *id_space_offset, size_t num_id_space_offset,
    const emb_t *embedding_vector, const uint32_t *embedding_vector_offset,
    const int *id_space_list, const int *local_id_space_list, size_t num_local_id_space_list,
    const key_t *key_location, const index_t *emb_table_id_space_offset, float *emb_table,
    const uint64_t *emb_table_ev_offset, const int *local_ev_size_list) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys) return;

  int id_space_idx = bs_upper_bound_sub_one(id_space_offset, num_id_space_offset, tid);
  assert(id_space_idx >= 0);
  int id_space = id_space_list[id_space_idx];

  int local_id_space_idx =
      bs_upper_bound_sub_one(local_id_space_list, num_local_id_space_list, id_space);
  assert(local_id_space_idx >= 0);
  index_t start = emb_table_id_space_offset[local_id_space_idx];
  index_t end = emb_table_id_space_offset[local_id_space_idx + 1];
  key_t k = keys[tid];

  int idx = bs_upper_bound_sub_one(key_location + start, end - start, k);
  assert(idx >= 0);

  uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
  int ev_size = local_ev_size_list[local_id_space_idx];

  const emb_t *ev_for_insert = embedding_vector + embedding_vector_offset[tid];
  for (int i = 0; i < ev_size; ++i) {
    float ei = HugeCTR::TypeConvertFunc<float, emb_t>::convert(ev_for_insert[i]);
    emb_table[ev_offset + idx * ev_size + i] = ei;
  }
}

template <typename key_t, typename index_t, typename emb_t>
__global__ void embedding_insert_by_tableindex_kernel(
    const key_t *insert_keys, size_t num_keys, const key_t *keys_table,
    const index_t *num_key_per_table_offset, const emb_t *insert_embedding_values,
    float *embedding_table, int table_index, size_t max_vocabulary_size,
    const uint64_t *embedding_table_offsets, const int *table_ev_size_list) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys) return;

  int embedding_vector_size = table_ev_size_list[table_index];
  key_t insert_key = insert_keys[tid];
  assert(insert_key < max_vocabulary_size);
  assert(insert_key >= 0);
  index_t key_offset = num_key_per_table_offset[table_index];
  uint64_t idx =
      static_cast<uint64_t>(bs_upper_bound_sub_one(keys_table + key_offset, num_keys, insert_key));
  uint64_t embedding_value_offset = embedding_table_offsets[table_index];
  float *tmp_embedding_table = embedding_table + embedding_value_offset;
  uint64_t input_offset = (uint64_t)tid * (uint64_t)embedding_vector_size;
  uint64_t output_offset = (uint64_t)idx * (uint64_t)embedding_vector_size;

  for (uint64_t i = 0; i < embedding_vector_size; ++i) {
    float ei =
        HugeCTR::TypeConvertFunc<float, emb_t>::convert(insert_embedding_values[input_offset + i]);
    tmp_embedding_table[output_offset + i] = ei;
  }
}

void RaggedStaticEmbeddingTable::assign(const core23::Tensor &keys, size_t num_keys,
                                        const core23::Tensor &num_unique_key_per_table_offset,
                                        size_t num_table_offset,
                                        const core23::Tensor &table_id_list,
                                        core23::Tensor &embeding_vector,
                                        const core23::Tensor &embedding_vector_offset) {
  CudaDeviceContext context(core_->get_device_id());

  DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION_CORE23(
        num_key_per_table_offset_.data_type().type(), index_t, [&] {
          auto stream = core_->get_local_gpu()->get_stream();

          {
            constexpr int block_size = 256;
            int grid_size = (static_cast<int64_t>(num_keys) - 1) / block_size + 1;
            embedding_insert_kernel<<<grid_size, block_size, 0, stream>>>(
                keys.data<key_t>(), num_keys, num_unique_key_per_table_offset.data<uint32_t>(),
                num_table_offset, embeding_vector.data<float>(),
                embedding_vector_offset.data<uint32_t>(), table_id_list.data<int>(),
                table_ids_.data<int>(), table_ids_.num_elements(), keys_.data<key_t>(),
                num_key_per_table_offset_.data<index_t>(), emb_table_.data<float>(),
                emb_table_ev_offset_.data<uint64_t>(), local_ev_size_list_.data<int>());
          }
        });
  });
}

void RaggedStaticEmbeddingTable::load(core23::Tensor &keys, core23::Tensor &id_space_offset,
                                      core23::Tensor &embedding_table, core23::Tensor &ev_size_list,
                                      core23::Tensor &id_space) {}

void RaggedStaticEmbeddingTable::dump(core23::Tensor *keys, core23::Tensor *id_space_offset,
                                      core23::Tensor *embedding_table, core23::Tensor *ev_size_list,
                                      core23::Tensor *id_space) {
  core23::Device device(core23::DeviceType::CPU);
  core23::TensorParams params = core23::TensorParams().device(device);

  *keys = core23::Tensor(params.shape({keys_.num_elements()}).data_type(keys_.data_type()));

  *id_space_offset = core23::Tensor(params.shape({num_key_per_table_offset_.num_elements()})
                                        .data_type(num_key_per_table_offset_.data_type()));

  *embedding_table =
      core23::Tensor(params.shape({emb_table_.num_elements()}).data_type(emb_table_.data_type()));

  *ev_size_list = core23::Tensor(params.shape({local_ev_size_list_.num_elements()})
                                     .data_type(local_ev_size_list_.data_type()));

  *id_space =
      core23::Tensor(params.shape({table_ids_.num_elements()}).data_type(table_ids_.data_type()));

  core23::copy_sync(*keys, keys_);
  core23::copy_sync(*id_space_offset, num_key_per_table_offset_);
  core23::copy_sync(*embedding_table, emb_table_);
  core23::copy_sync(*ev_size_list, local_ev_size_list_);
  core23::copy_sync(*id_space, table_ids_);
}

void RaggedStaticEmbeddingTable::dump_by_id(core23::Tensor *h_keys_tensor,
                                            core23::Tensor *h_embedding_table, int table_id) {
  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = keys_.data_type();
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    key_t *d_keys = (key_t *)keys_.data();
    d_keys += h_num_key_per_table_offset_[table_index];
    key_t *h_keys = (key_t *)h_keys_tensor->data();
    HCTR_LIB_THROW(cudaMemcpy(h_keys, d_keys, sizeof(key_t) * h_num_key_per_table_[table_index],
                              cudaMemcpyDeviceToHost));

    float *d_embedding_vector = (float *)emb_table_.data();
    d_embedding_vector += h_emb_table_ev_offset_[table_index];
    float *h_embedding_vector = (float *)h_embedding_table->data();
    HCTR_LIB_THROW(cudaMemcpy(h_embedding_vector, d_embedding_vector,
                              sizeof(float) * h_size_per_table_[table_index],
                              cudaMemcpyDeviceToHost));
  });
}

void RaggedStaticEmbeddingTable::load_by_id(core23::Tensor *h_keys_tensor,
                                            core23::Tensor *h_embedding_table, int table_id) {
  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = keys_.data_type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION_CORE23(
        num_key_per_table_offset_.data_type().type(), index_t, [&] {
          core23::Device device(core23::DeviceType::GPU, core_->get_device_id());
          core23::TensorParams params = core23::TensorParams().device(device);

          auto d_keys =
              core23::Tensor(params.shape({h_keys_tensor->num_elements()}).data_type(key_type));
          auto d_embedding_vector = core23::Tensor(params.shape({h_embedding_table->num_elements()})
                                                       .data_type(core23::ScalarType::Float));

          core23::copy_sync(d_keys, *h_keys_tensor);
          core23::copy_sync(d_embedding_vector, *h_embedding_table);
          size_t max_vocabulary_size = h_table_max_vocabulary_size_[table_index];
          size_t num_keys = h_keys_tensor->num_elements();
          size_t table_keys = h_num_key_per_table_[table_index];

          {
            constexpr int block_size = 256;
            int grid_size =
                (static_cast<int64_t>(h_keys_tensor->num_elements()) - 1) / block_size + 1;
            embedding_insert_by_tableindex_kernel<<<grid_size, block_size>>>(
                (key_t *)d_keys.data(), num_keys, keys_.data<key_t>(),
                num_key_per_table_offset_.data<index_t>(), (float *)d_embedding_vector.data(),
                emb_table_.data<float>(), table_index, max_vocabulary_size,
                emb_table_ev_offset_.data<uint64_t>(), local_ev_size_list_.data<int>());
          }
        });
  });
}

size_t RaggedStaticEmbeddingTable::size() const { return emb_table_size_; }

size_t RaggedStaticEmbeddingTable::capacity() const { return emb_table_size_; }

size_t RaggedStaticEmbeddingTable::key_num() const {
  return accumulate(h_num_key_per_table_.begin(), h_num_key_per_table_.end(), 0);
}

std::vector<size_t> RaggedStaticEmbeddingTable::size_per_table() const { return h_size_per_table_; }

std::vector<size_t> RaggedStaticEmbeddingTable::capacity_per_table() const {
  return h_size_per_table_;
}

std::vector<size_t> RaggedStaticEmbeddingTable::key_num_per_table() const {
  return h_num_key_per_table_;
}

std::vector<int> RaggedStaticEmbeddingTable::table_ids() const { return h_table_ids_; }

std::vector<int> RaggedStaticEmbeddingTable::table_evsize() const { return h_local_ev_sizes_; };

void RaggedStaticEmbeddingTable::clear() {}
}  // namespace embedding
