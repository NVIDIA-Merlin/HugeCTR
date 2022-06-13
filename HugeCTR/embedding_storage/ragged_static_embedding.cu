/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <utils.cuh>

#include "HugeCTR/core/registry.hpp"
#include "HugeCTR/embedding/view.hpp"
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/utils.cuh"
#include "ragged_static_embedding.hpp"

namespace embedding {

namespace {

template <typename key_t, typename index_t>
__global__ void ragged_static_embedding_table_lookup_kernel(
    const key_t *key, size_t num_keys, const uint32_t *id_space_offset, size_t num_id_space_offset,
    const int *id_space_list, const int *local_id_space_list, size_t num_local_id_space_list,
    const key_t *key_location, const index_t *emb_table_id_space_offset, float *emb_table,
    const uint64_t *emb_table_ev_offset, const int *local_ev_size_list, float **emb_vec) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys) return;
  int id_space_idx = binary_search_index_lower_bound(id_space_offset, num_id_space_offset, tid);
  assert(id_space_idx >= 0);
  int id_space = id_space_list[id_space_idx];

  int local_id_space_idx =
      binary_search_index_lower_bound(local_id_space_list, num_local_id_space_list, id_space);
  assert(local_id_space_idx >= 0);
  index_t start = emb_table_id_space_offset[local_id_space_idx];
  index_t end = emb_table_id_space_offset[local_id_space_idx + 1];
  key_t k = key[tid];

  int idx = binary_search_index_lower_bound(key_location + start, end - start, k);
  assert(idx >= 0);

  uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
  int ev_size = local_ev_size_list[local_id_space_idx];

  emb_vec[tid] = &emb_table[ev_offset + idx * ev_size];
}

__global__ void sgd_update_grad_kernel(const uint32_t *ev_offset, size_t num_ev, float lr,
                                       float scaler, float *grad_ev) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_ev) return;
  uint64_t start = ev_offset[tid];
  uint64_t end = ev_offset[tid + 1];

  for (uint32_t i = start; i < end; ++i) {
    float gi = grad_ev[i] / scaler;
    grad_ev[i] = (-lr * gi);
  }
}

template <typename key_t, typename index_t, typename emb_t>
__global__ void update_kernel(const key_t *keys, size_t num_keys, const uint32_t *id_space_offset,
                              size_t num_id_space_offset, const emb_t *grad_ev,
                              const uint32_t *grad_ev_offset, const int *id_space_list,
                              const int *local_id_space_list, size_t num_local_id_space_list,
                              const key_t *key_location, const index_t *emb_table_id_space_offset,
                              float *emb_table, const uint64_t *emb_table_ev_offset,
                              const int *local_ev_size_list) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys) return;

  int id_space_idx = binary_search_index_lower_bound(id_space_offset, num_id_space_offset, tid);
  assert(id_space_idx >= 0);
  int id_space = id_space_list[id_space_idx];

  int local_id_space_idx =
      binary_search_index_lower_bound(local_id_space_list, num_local_id_space_list, id_space);
  assert(local_id_space_idx >= 0);
  index_t start = emb_table_id_space_offset[local_id_space_idx];
  index_t end = emb_table_id_space_offset[local_id_space_idx + 1];
  key_t k = keys[tid];

  int idx = binary_search_index_lower_bound(key_location + start, end - start, k);
  assert(idx >= 0);

  uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
  int ev_size = local_ev_size_list[local_id_space_idx];

  const emb_t *grad_ev_for_update = grad_ev + grad_ev_offset[tid];
  for (int i = 0; i < ev_size; ++i) {
    float gi = HugeCTR::TypeConvertFunc<float, emb_t>::convert(grad_ev_for_update[i]);
    emb_table[ev_offset + idx * ev_size + i] += gi;
  }
}
}  // namespace

RaggedStaticEmbeddingTable::RaggedStaticEmbeddingTable(
    const HugeCTR::GPUResource &gpu_resource, std::shared_ptr<CoreResourceManager> core,
    const std::vector<EmbeddingTableParam> &global_emb_table_param_list,
    const EmbeddingCollectionParam &ebc_param, const EmbeddingShardingParam &sharding_param,
    const HugeCTR::OptParams &opt_param)
    : core_(core), emb_table_size_(0), opt_param_(opt_param) {
  CudaDeviceContext ctx(core_->get_device_id());
  auto key_type = ebc_param.key_type;
  auto index_type = ebc_param.index_type;

  std::set<int> id_space_set;
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(index_type.type(), index_t, [&] {
      std::vector<int> cpu_local_id_space_list;
      std::vector<key_t> cpu_key_list;
      std::vector<index_t> cpu_id_space_offset{0};

      std::vector<uint64_t> cpu_emb_table_ev_offset{0};
      std::vector<int> cpu_local_ev_size_list;

      std::set<int> id_space_set;
      for (int emb_id : sharding_param.local_embedding_list) {
        int id_space = ebc_param.embedding_params[emb_id].id_space;
        auto &emb_table_param = global_emb_table_param_list[id_space];

        if (id_space_set.find(id_space) == id_space_set.end()) {
          uint64_t id_space_count = 0;
          for (int64_t k = emb_table_param.min_key; k < emb_table_param.max_key; ++k) {
            if (k % sharding_param.num_sharding == sharding_param.sharding_id) {
              cpu_key_list.push_back(k);
              id_space_count += 1;
            }
          }
          cpu_local_id_space_list.push_back(id_space);
          cpu_id_space_offset.push_back(id_space_count);

          uint64_t segment_emb_table_size = id_space_count * emb_table_param.ev_size;
          cpu_emb_table_ev_offset.push_back(segment_emb_table_size);
          cpu_local_ev_size_list.push_back(emb_table_param.ev_size);
          emb_table_size_ += segment_emb_table_size;
        }
        id_space_set.insert(id_space);
      }

      std::partial_sum(cpu_id_space_offset.begin(), cpu_id_space_offset.end(),
                       cpu_id_space_offset.begin());
      std::partial_sum(cpu_emb_table_ev_offset.begin(), cpu_emb_table_ev_offset.end(),
                       cpu_emb_table_ev_offset.begin());

      auto buffer_ptr = GetBuffer(core);
      local_id_space_list_ = buffer_ptr->reserve(cpu_local_id_space_list.size(), DeviceType::GPU,
                                                 TensorScalarType::Int32);
      key_location_ = buffer_ptr->reserve(cpu_key_list.size(), DeviceType::GPU, key_type);
      id_space_offset_ =
          buffer_ptr->reserve(cpu_id_space_offset.size(), DeviceType::GPU, index_type);
      emb_table_ = buffer_ptr->reserve(emb_table_size_, DeviceType::GPU, TensorScalarType::Float32);
      emb_table_ev_offset_ = buffer_ptr->reserve(cpu_emb_table_ev_offset.size(), DeviceType::GPU,
                                                 TensorScalarType::UInt64);
      local_ev_size_list_ = buffer_ptr->reserve(cpu_local_ev_size_list.size(), DeviceType::GPU,
                                                TensorScalarType::Int32);
      buffer_ptr->allocate();

      local_id_space_list_.copy_from(cpu_local_id_space_list);
      key_location_.copy_from(cpu_key_list);
      id_space_offset_.copy_from(cpu_id_space_offset);
      emb_table_ev_offset_.copy_from(cpu_emb_table_ev_offset);
      local_ev_size_list_.copy_from(cpu_local_ev_size_list);

      auto uniform_init_table = [&](const curandGenerator_t &generator) {
        const size_t num_tables = cpu_local_id_space_list.size();
        for (size_t embedding = 0; embedding < num_tables; embedding++) {
          index_t num_keys = cpu_id_space_offset[embedding + 1] - cpu_id_space_offset[embedding];
          float up_bound = sqrt(1.f / num_keys);
          size_t offset = cpu_emb_table_ev_offset[embedding];
          size_t num_elements =
              cpu_emb_table_ev_offset[embedding + 1] - cpu_emb_table_ev_offset[embedding];

          HugeCTR::UniformGenerator::fill(emb_table_.get<float>() + offset, num_elements,
          -up_bound,
                                          up_bound, gpu_resource.get_sm_count(), generator,
                                          gpu_resource.get_stream());
        }
      };
      if (sharding_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
        uniform_init_table(gpu_resource.get_replica_uniform_curand_generator());
      } else {
        uniform_init_table(gpu_resource.get_replica_variant_curand_generator());
      }
    });
  });
}

void RaggedStaticEmbeddingTable::lookup(const Tensor &keys, size_t num_keys,
                                        const Tensor &id_space_offset, size_t num_id_space_offset,
                                        const Tensor &id_space_list, TensorList &emb_vec) {
  CudaDeviceContext ctx(core_->get_device_id());

  DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(id_space_offset_.dtype().type(), index_t, [&] {
      cudaStream_t stream = core_->get_local_gpu()->get_stream();

      if (num_keys > 0) {  // batch size is small there can be situation that we do not need have
                           // key for lookup
        constexpr int block_size = 256;
        int grid_size = (num_keys - 1) / block_size + 1;
        ragged_static_embedding_table_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            keys.get<key_t>(), num_keys, id_space_offset.get<uint32_t>(), num_id_space_offset,
            id_space_list.get<int>(), local_id_space_list_.get<int>(),
            local_id_space_list_.get_num_elements(), key_location_.get<key_t>(),
            id_space_offset_.get<index_t>(), emb_table_.get<float>(),
            emb_table_ev_offset_.get<uint64_t>(), local_ev_size_list_.get<int>(),
            emb_vec.get<float>());
      }

      HCTR_LIB_THROW(cudaPeekAtLastError());
    });
  });
}

void RaggedStaticEmbeddingTable::update(const Tensor &keys, size_t num_keys,
                                        const Tensor &id_space_offset, size_t num_id_space_offset,
                                        const Tensor &id_space_list, Tensor &grad_ev,
                                        const Tensor &grad_ev_offset) {
  CudaDeviceContext context(core_->get_device_id());

  HCTR_CHECK_HINT(opt_param_.optimizer != HugeCTR::Optimizer_t::NOT_INITIALIZED,
                  "optimizer not initialized");

  DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(id_space_offset_.dtype().type(), index_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();

      if (opt_param_.optimizer == HugeCTR::Optimizer_t::SGD) {
        constexpr int block_size = 256;
        int grid_size = (static_cast<int64_t>(num_keys) - 1) / block_size + 1;
        sgd_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_ev_offset.get<uint32_t>(), num_keys, opt_param_.lr, opt_param_.scaler,
            grad_ev.get<float>());
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
      }

      {
        constexpr int block_size = 256;
        int grid_size = (static_cast<int64_t>(num_keys) - 1) / block_size + 1;
        update_kernel<<<grid_size, block_size, 0, stream>>>(
            keys.get<key_t>(), num_keys, id_space_offset.get<uint32_t>(), num_id_space_offset,
            grad_ev.get<float>(), grad_ev_offset.get<uint32_t>(), id_space_list.get<int>(),
            local_id_space_list_.get<int>(), local_id_space_list_.get_num_elements(),
            key_location_.get<key_t>(), id_space_offset_.get<index_t>(), emb_table_.get<float>(),
            emb_table_ev_offset_.get<uint64_t>(), local_ev_size_list_.get<int>());
      }
    });
  });
}

void RaggedStaticEmbeddingTable::load(Tensor &keys, Tensor &id_space_offset,
                                      Tensor &embedding_table, Tensor &ev_size_list,
                                      Tensor &id_space) {}

void RaggedStaticEmbeddingTable::dump(Tensor *keys, Tensor *id_space_offset,
                                      Tensor *embedding_table, Tensor *ev_size_list,
                                      Tensor *id_space) {
  Device device{DeviceType::CPU};

  *keys = key_location_.to(core_, device);
  *id_space_offset = id_space_offset_.to(core_, device);
  *embedding_table = emb_table_.to(core_, device);
  *ev_size_list = local_ev_size_list_.to(core_, device);
  *id_space = local_id_space_list_.to(core_, device);
}

size_t RaggedStaticEmbeddingTable::size() const { return emb_table_size_; }

size_t RaggedStaticEmbeddingTable::capacity() const { return emb_table_size_; }

void RaggedStaticEmbeddingTable::clear() {}

}  // namespace embedding
