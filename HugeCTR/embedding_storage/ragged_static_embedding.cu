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

#include <core/registry.hpp>
#include <data_simulator.hpp>
#include <embedding/view.hpp>
#include <embedding_storage/ragged_static_embedding.hpp>
#include <numeric>
#include <utils.cuh>

namespace embedding {

namespace {

template <typename key_t, typename index_t>
__global__ void ragged_static_embedding_table_lookup_kernel(
    const key_t *key, size_t num_keys, const uint32_t *id_space_offset, size_t num_id_space_offset,
    const int *id_space_list, const int *local_id_space_list, size_t num_local_id_space_list,
    const key_t *key_location, const index_t *emb_table_id_space_offset, float *emb_table,
    const uint64_t *emb_table_ev_offset, const int *local_ev_size_list, float **emb_vec) {
  for (uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_keys;
       tid += blockDim.x * gridDim.x) {
    int id_space_idx = binary_search_index_lower_bound(id_space_offset, num_id_space_offset, tid);
    assert(id_space_idx >= 0);
    assert(id_space_idx < num_id_space_offset);

    int id_space = id_space_list[id_space_idx];

    int local_id_space_idx =
        binary_search_index_lower_bound(local_id_space_list, num_local_id_space_list, id_space);
    assert(local_id_space_idx >= 0);
    assert(local_id_space_idx < num_local_id_space_list);

    index_t start = emb_table_id_space_offset[local_id_space_idx];
    index_t end = emb_table_id_space_offset[local_id_space_idx + 1];
    key_t k = key[tid];

    // Attention: we must convert idx to uint64_t so when we multiply idx with ev_size it would get
    // overflow. So as in update_kernel
    uint64_t idx = static_cast<uint64_t>(
        binary_search_index_lower_bound(key_location + start, end - start, k));
    assert(idx >= 0);
    assert(idx < static_cast<uint64_t>(end - start));

    uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
    int ev_size = local_ev_size_list[local_id_space_idx];
    assert(static_cast<uint64_t>(ev_offset + idx * ev_size) <
           emb_table_ev_offset[local_id_space_idx + 1]);

    emb_vec[tid] = &emb_table[ev_offset + idx * ev_size];
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
  for (uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_keys;
       tid += blockDim.x * gridDim.x) {
    int id_space_idx = binary_search_index_lower_bound(id_space_offset, num_id_space_offset, tid);
    assert(id_space_idx >= 0);
    assert(id_space_idx < num_id_space_offset);
    int id_space = id_space_list[id_space_idx];

    int local_id_space_idx =
        binary_search_index_lower_bound(local_id_space_list, num_local_id_space_list, id_space);
    assert(local_id_space_idx >= 0);
    assert(local_id_space_idx < num_local_id_space_list);
    index_t start = emb_table_id_space_offset[local_id_space_idx];
    index_t end = emb_table_id_space_offset[local_id_space_idx + 1];
    key_t k = keys[tid];

    uint64_t idx = static_cast<uint64_t>(
        binary_search_index_lower_bound(key_location + start, end - start, k));
    assert(idx >= 0);
    assert(idx < end - start);

    uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
    int ev_size = local_ev_size_list[local_id_space_idx];
    assert(ev_offset + idx * ev_size < emb_table_ev_offset[local_id_space_idx + 1]);

    const emb_t *grad_ev_for_update = grad_ev + grad_ev_offset[tid];
    for (int i = 0; i < ev_size; ++i) {
      float gi = HugeCTR::TypeConvertFunc<float, emb_t>::convert(grad_ev_for_update[i]);
      emb_table[ev_offset + idx * ev_size + i] += gi;
    }
  }
}

template <typename key_t, typename index_t>
struct RaggedKeyToIndicesFunc {
  int *local_table_ids;
  int *local_ev_sizes;
  int64_t num_local_table_ids;

  key_t *key_location;
  index_t *emb_table_id_space_offset;
  uint64_t *emb_table_ev_start_indices;

  DEVICE_INLINE void operator()(const key_t &key, const int &table_id,
                                uint64_t *ev_start_indices_ptr, int *ev_size_ptr) {
    int local_id_space_idx =
        binary_search_index_lower_bound(local_table_ids, num_local_table_ids, table_id);
    assert(local_id_space_idx >= 0);
    assert(local_id_space_idx < num_local_table_ids);
    index_t start = emb_table_id_space_offset[local_id_space_idx];
    index_t end = emb_table_id_space_offset[local_id_space_idx + 1];

    uint64_t idx = static_cast<uint64_t>(
        binary_search_index_lower_bound(key_location + start, end - start, key));
    assert(idx >= 0);
    assert(idx < end - start);

    uint64_t ev_offset = emb_table_ev_start_indices[local_id_space_idx];
    int ev_size = local_ev_sizes[local_id_space_idx];
    assert(ev_offset + idx * ev_size < emb_table_ev_start_indices[local_id_space_idx + 1]);
    *ev_start_indices_ptr = ev_offset + idx * static_cast<uint64_t>(ev_size);
    *ev_size_ptr = ev_size;
  }
};

template <typename key_t, typename emb_t>
struct OptimizierInput {
  const emb_t *wgrad;
  uint64_t ev_start_indices;
  int ev_id;
  float lr;
  float scaler;
};

template <typename key_t, typename emb_t>
struct SGDOptimizer {
  DEVICE_INLINE float update(const OptimizierInput<key_t, emb_t> &input) {
    return -input.lr * (HugeCTR::TypeConvertFunc<float, emb_t>::convert(input.wgrad[input.ev_id]) /
                        input.scaler);
  }
};

template <typename key_t, typename emb_t, typename acc_t, typename KeyToIndicesFunc>
struct AdaGradOptimizer {
  KeyToIndicesFunc key_to_indices_func;
  acc_t *v;
  float epsilon;

  DEVICE_INLINE float update(const OptimizierInput<key_t, emb_t> &input) {
    float vi =
        HugeCTR::TypeConvertFunc<float, acc_t>::convert(v[input.ev_start_indices + input.ev_id]);
    float gi = HugeCTR::TypeConvertFunc<float, emb_t>::convert(input.wgrad[input.ev_id]);
    gi = gi / input.scaler;
    vi = vi + gi * gi;

    gi = -input.lr * gi / (sqrtf(vi) + epsilon);
    v[input.ev_start_indices + input.ev_id] = HugeCTR::TypeConvertFunc<acc_t, float>::convert(vi);
    return gi;
  }
};

template <typename key_t, typename index_t, typename emb_t, typename OptimizerFunc,
          typename KeyToIndicesFunc>
__global__ void update_kernel(const key_t *keys, const size_t *num_keys_ptr, const int *table_ids,
                              const emb_t *grad_ev, const uint32_t *ev_start_indices,
                              KeyToIndicesFunc key_to_indices_func, float *emb_table,
                              OptimizerFunc optimizer, float lr, float scaler) {
  size_t num_steps = (*num_keys_ptr - 1) / (blockDim.x * gridDim.x) + 1;
  for (size_t step = 0; step < num_steps; step++) {
    size_t tid = step * blockDim.x * gridDim.x + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t emb_table_ev_start_indices_frag;
    int ev_size_frag = std::numeric_limits<int>::max();
    uint32_t grad_ev_offset_frag;
    if (tid < *num_keys_ptr) {
      key_t key = keys[tid];
      int table_id = table_ids[tid];
      key_to_indices_func(key, table_id, &emb_table_ev_start_indices_frag, &ev_size_frag);
      grad_ev_offset_frag = ev_start_indices[tid];
    }

    for (int lane_id = 0; lane_id < warpSize; lane_id++) {
      int ev_size = __shfl_sync(0xffffffff, ev_size_frag, lane_id);
      if (ev_size == std::numeric_limits<int>::max()) {
        break;
      }
      const emb_t *grad_ev_for_update =
          grad_ev + __shfl_sync(0xffffffff, grad_ev_offset_frag, lane_id);
      uint64_t ev_start_indices = __shfl_sync(0xffffffff, emb_table_ev_start_indices_frag, lane_id);
      float *ev = emb_table + ev_start_indices;

      for (int i = threadIdx.x % warpSize; i < ev_size; i += warpSize) {
        OptimizierInput<key_t, emb_t> input{grad_ev_for_update, ev_start_indices, i, lr, scaler};
        float gi = optimizer.update(input);
        ev[i] += gi;
      }
    }
  }
}

template <typename key_t, typename index_t, typename emb_t>
__global__ void embedding_insert_kernel(
    const key_t *keys, size_t num_keys, const uint32_t *id_space_offset, size_t num_id_space_offset,
    const emb_t *embedding_vector, const uint32_t *embedding_vector_offset,
    const int *id_space_list, const int *local_id_space_list, size_t num_local_id_space_list,
    const key_t *key_location, const index_t *emb_table_id_space_offset, float *emb_table,
    const uint64_t *emb_table_ev_offset, const int *local_ev_size_list) {
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
  uint64_t idx = static_cast<uint64_t>(
      binary_search_index_lower_bound(keys_table + key_offset, num_keys, insert_key));
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

}  // namespace

RaggedStaticEmbeddingTable::RaggedStaticEmbeddingTable(
    const HugeCTR::GPUResource &gpu_resource, std::shared_ptr<CoreResourceManager> core,
    const std::vector<EmbeddingTableParam> &table_params, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id, const HugeCTR::OptParams &opt_param)
    : core_(core), emb_table_size_(0), opt_param_(opt_param) {
  CudaDeviceContext ctx(core_->get_device_id());
  int global_gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();
  HCTR_CHECK_HINT(num_gpus == static_cast<int>(ebc_param.shard_matrix.size()),
                  "num_gpus is not match with shard matrix");

  auto key_type = ebc_param.key_type;
  auto index_type = ebc_param.index_type;
  auto emb_type = ebc_param.emb_type;

  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(index_type.type(), index_t, [&] {
      std::vector<key_t> h_key_list;
      std::vector<index_t> h_num_key_per_table_offset{0};
      h_emb_table_ev_offset_.push_back(0);
      const auto &emb_param = ebc_param.grouped_emb_params[grouped_id];
      if (emb_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
        for (int table_id : emb_param.table_ids) {
          uint64_t num_key = 0;
          h_table_ids_.push_back(table_id);
          h_table_max_vocabulary_size_.push_back(table_params[table_id].max_vocabulary_size);
          for (int64_t k = 0; k < table_params[table_id].max_vocabulary_size; ++k) {
            h_key_list.push_back(k);
            num_key += 1;
          }
          h_num_key_per_table_.push_back(num_key);
          h_num_key_per_table_offset.push_back(num_key);

          uint64_t segment_emb_table_size = num_key * table_params[table_id].ev_size;
          h_size_per_table_.push_back(segment_emb_table_size);
          h_emb_table_ev_offset_.push_back(segment_emb_table_size);
          h_local_ev_sizes_.push_back(table_params[table_id].ev_size);
          emb_table_size_ += segment_emb_table_size;
        }
      } else if (emb_param.table_placement_strategy == TablePlacementStrategy::ModelParallel) {
        for (int table_id : emb_param.table_ids) {
          std::vector<int> shard_gpu_list;
          for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            HCTR_CHECK_HINT(table_id < static_cast<int>(ebc_param.shard_matrix[gpu_id].size()),
                            "table_id is out of range");
            if (ebc_param.shard_matrix[gpu_id][table_id] == 1) {
              shard_gpu_list.push_back(gpu_id);
            }
          }
          int num_shards = static_cast<int>(shard_gpu_list.size());
          auto find_shard_id_iter =
              std::find(shard_gpu_list.begin(), shard_gpu_list.end(), global_gpu_id);
          if (find_shard_id_iter == shard_gpu_list.end()) {
            continue;
          }
          uint64_t num_key = 0;
          h_table_ids_.push_back(table_id);
          h_table_max_vocabulary_size_.push_back(table_params[table_id].max_vocabulary_size);
          int shard_id =
              static_cast<int>(std::distance(shard_gpu_list.begin(), find_shard_id_iter));
          for (int64_t k = 0; k < table_params[table_id].max_vocabulary_size; ++k) {
            if (k % num_shards == shard_id) {
              h_key_list.push_back(k);
              num_key += 1;
            }
          }

          h_num_key_per_table_.push_back(num_key);
          h_num_key_per_table_offset.push_back(num_key);
          uint64_t segment_emb_table_size = num_key * table_params[table_id].ev_size;
          h_size_per_table_.push_back(segment_emb_table_size);
          h_emb_table_ev_offset_.push_back(segment_emb_table_size);
          h_local_ev_sizes_.push_back(table_params[table_id].ev_size);
          emb_table_size_ += segment_emb_table_size;
        }
      }

      std::partial_sum(h_num_key_per_table_offset.begin(), h_num_key_per_table_offset.end(),
                       h_num_key_per_table_offset.begin());
      std::partial_sum(h_emb_table_ev_offset_.begin(), h_emb_table_ev_offset_.end(),
                       h_emb_table_ev_offset_.begin());
      for (auto tmp_offset : h_num_key_per_table_offset) {
        h_num_key_per_table_offset_.push_back(static_cast<size_t>(tmp_offset));
      }

      auto buffer_ptr = GetBuffer(core);
      table_ids_ =
          buffer_ptr->reserve(h_table_ids_.size(), DeviceType::GPU, TensorScalarType::Int32);
      keys_ = buffer_ptr->reserve(h_key_list.size(), DeviceType::GPU, key_type);
      num_key_per_table_offset_ =
          buffer_ptr->reserve(h_num_key_per_table_offset.size(), DeviceType::GPU, index_type);
      emb_table_ = buffer_ptr->reserve(emb_table_size_, DeviceType::GPU, TensorScalarType::Float32);
      emb_table_ev_offset_ = buffer_ptr->reserve(h_emb_table_ev_offset_.size(), DeviceType::GPU,
                                                 TensorScalarType::UInt64);
      local_ev_size_list_ =
          buffer_ptr->reserve(h_local_ev_sizes_.size(), DeviceType::GPU, TensorScalarType::Int32);
      buffer_ptr->allocate();
      table_ids_.copy_from(h_table_ids_);
      keys_.copy_from(h_key_list);
      num_key_per_table_offset_.copy_from(h_num_key_per_table_offset);
      emb_table_ev_offset_.copy_from(h_emb_table_ev_offset_);
      local_ev_size_list_.copy_from(h_local_ev_sizes_);
      if (opt_param.optimizer == HugeCTR::Optimizer_t::AdaGrad) {
        DISPATCH_FLOAT_AND_HALF_FUNCTION(emb_type.type(), emb_t, [&] {
          auto accum_tensor =
              buffer_ptr->reserve(emb_table_size_, DeviceType::GPU, TensorScalarType::Float32);
          buffer_ptr->allocate();
          HCTR_LIB_THROW(cudaMemset(accum_tensor.get(), 0, accum_tensor.nbytes()));
          opt_buffer_ = AdaGradOptBuffer{accum_tensor};
        });
      }

      for (size_t i = 0; i < h_table_ids_.size(); i++) {
        int table_id = h_table_ids_[i];
        std::function<void(const curandGenerator_t &)> init_table_functor;

        if (table_params[table_id].init_param.initializer_type == HugeCTR::Initializer_t::Default) {
          init_table_functor = [&](const curandGenerator_t &generator) {
            float up_bound = sqrt(1.f / h_table_max_vocabulary_size_[i]);
            size_t offset = h_emb_table_ev_offset_[i];
            size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

            HugeCTR::UniformGenerator::fill(emb_table_.get<float>() + offset, num_elements,
                                            -up_bound, up_bound, gpu_resource.get_sm_count(),
                                            generator, gpu_resource.get_stream());
          };
        } else if (table_params[table_id].init_param.initializer_type ==
                   HugeCTR::Initializer_t::Uniform) {
          init_table_functor = [&](const curandGenerator_t &generator) {
            float up_bound = table_params[table_id].init_param.uniform_params.up_bound;
            size_t offset = h_emb_table_ev_offset_[i];
            size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

            HugeCTR::UniformGenerator::fill(emb_table_.get<float>() + offset, num_elements,
                                            -up_bound, up_bound, gpu_resource.get_sm_count(),
                                            generator, gpu_resource.get_stream());
          };
        } else if (table_params[table_id].init_param.initializer_type ==
                   HugeCTR::Initializer_t::Sinusoidal) {
          init_table_functor = [&](const curandGenerator_t &) {
            const SinusoidalParams &sinus_params =
                table_params[table_id].init_param.sinusoidal_params;
            int max_sequence_len = sinus_params.max_sequence_len;
            int ev_size = sinus_params.ev_size;
            size_t offset = h_emb_table_ev_offset_[i];
            size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

            HCTR_CHECK_HINT(max_sequence_len * ev_size == static_cast<int>(num_elements),
                            "max_sequent_len * ev_size %d should equal to num_elements %d",
                            max_sequence_len * ev_size, static_cast<int>(num_elements));
            HugeCTR::SinusoidalGenerator::fill(
                emb_table_.get<float>() + offset, num_elements, ev_size, max_sequence_len,
                gpu_resource.get_sm_count(), gpu_resource.get_stream());
          };
        } else {
          HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "initializer not implemented");
        }

        // data parallel table should use same curand seed across all gpus
        if (emb_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
          init_table_functor(gpu_resource.get_replica_uniform_curand_generator());
        } else {
          init_table_functor(gpu_resource.get_replica_variant_curand_generator());
        }
      }
    });
  });
}

void RaggedStaticEmbeddingTable::lookup(const Tensor &keys, size_t num_keys,
                                        const Tensor &id_space_offset, size_t num_id_space_offset,
                                        const Tensor &id_space_list, TensorList &emb_vec) {
  CudaDeviceContext ctx(core_->get_device_id());

  DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(num_key_per_table_offset_.dtype().type(), index_t, [&] {
      cudaStream_t stream = core_->get_local_gpu()->get_stream();

      if (num_keys > 0) {  // batch size is small there can be situation that we do not need have
                           // key for lookup
        constexpr int block_size = 256;
        int grid_size = (num_keys - 1) / block_size + 1;
        ragged_static_embedding_table_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            keys.get<key_t>(), num_keys, id_space_offset.get<uint32_t>(), num_id_space_offset,
            id_space_list.get<int>(), table_ids_.get<int>(), table_ids_.get_num_elements(),
            keys_.get<key_t>(), num_key_per_table_offset_.get<index_t>(), emb_table_.get<float>(),
            emb_table_ev_offset_.get<uint64_t>(), local_ev_size_list_.get<int>(),
            emb_vec.get<float>());
      }

      HCTR_LIB_THROW(cudaPeekAtLastError());
    });
  });
}

void RaggedStaticEmbeddingTable::update(const Tensor &unique_keys, const Tensor &num_unique_keys,
                                        const Tensor &table_ids, const Tensor &ev_start_indices,
                                        const Tensor &wgrad) {
  CudaDeviceContext context(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK_HINT(opt_param_.optimizer != HugeCTR::Optimizer_t::NOT_INITIALIZED,
                  "optimizer not initialized");
  HCTR_CHECK(num_unique_keys.dtype() == core::TensorScalarType::Size_t);
  HCTR_CHECK(table_ids.dtype() == core::TensorScalarType::Int32);
  HCTR_CHECK(ev_start_indices.dtype() == core::TensorScalarType::UInt32);
  HCTR_CHECK(wgrad.dtype() == core::TensorScalarType::Float32);

  if (opt_param_.optimizer == HugeCTR::Optimizer_t::SGD) {
    DISPATCH_INTEGRAL_FUNCTION(unique_keys.dtype().type(), key_t, [&] {
      DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(num_key_per_table_offset_.dtype().type(), index_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION(wgrad.dtype().type(), emb_t, [&] {
          RaggedKeyToIndicesFunc<key_t, index_t> key_to_indices_func{
              table_ids_.get<int>(),
              local_ev_size_list_.get<int>(),
              table_ids_.get_num_elements(),
              keys_.get<key_t>(),
              num_key_per_table_offset_.get<index_t>(),
              emb_table_ev_offset_.get<uint64_t>(),
          };
          SGDOptimizer<key_t, emb_t> optimizer;

          constexpr int block_size = 256;
          constexpr int grid_size = 144 * 8;
          update_kernel<key_t, index_t, emb_t><<<grid_size, block_size, 0, stream>>>(
              unique_keys.get<key_t>(), num_unique_keys.get<size_t>(), table_ids.get<int>(),
              wgrad.get<emb_t>(), ev_start_indices.get<uint32_t>(), key_to_indices_func,
              emb_table_.get<float>(), optimizer, opt_param_.lr, opt_param_.scaler);
        });
      });
    });
  } else if (opt_param_.optimizer == HugeCTR::Optimizer_t::AdaGrad) {
    DISPATCH_INTEGRAL_FUNCTION(unique_keys.dtype().type(), key_t, [&] {
      DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(num_key_per_table_offset_.dtype().type(), index_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION(wgrad.dtype().type(), emb_t, [&] {
          auto adagrad_opt_buffer = std::get_if<AdaGradOptBuffer>(&opt_buffer_);
          HCTR_CHECK_HINT(adagrad_opt_buffer != nullptr, "Adagrad Opt Buffer not initialized.");
          DISPATCH_FLOAT_AND_HALF_FUNCTION(
              adagrad_opt_buffer->opt_accum_tensor.dtype().type(), acc_t, [&] {
                RaggedKeyToIndicesFunc<key_t, index_t> key_to_indices_func{
                    table_ids_.get<int>(),
                    local_ev_size_list_.get<int>(),
                    table_ids_.get_num_elements(),
                    keys_.get<key_t>(),
                    num_key_per_table_offset_.get<index_t>(),
                    emb_table_ev_offset_.get<uint64_t>(),
                };
                AdaGradOptimizer<key_t, emb_t, acc_t, decltype(key_to_indices_func)> optimizer{
                    key_to_indices_func, adagrad_opt_buffer->opt_accum_tensor.get<acc_t>(),
                    opt_param_.hyperparams.adagrad.epsilon};

                constexpr int block_size = 256;
                constexpr int grid_size = 8 * 144;
                update_kernel<key_t, index_t, emb_t><<<grid_size, block_size, 0, stream>>>(
                    unique_keys.get<key_t>(), num_unique_keys.get<size_t>(), table_ids.get<int>(),
                    wgrad.get<emb_t>(), ev_start_indices.get<uint32_t>(), key_to_indices_func,
                    emb_table_.get<float>(), optimizer, opt_param_.lr, opt_param_.scaler);
              });
        });
      });
    });
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
  }
}

void RaggedStaticEmbeddingTable::assign(const Tensor &keys, size_t num_keys,
                                        const Tensor &num_unique_key_per_table_offset,
                                        size_t num_table_offset, const Tensor &table_id_list,
                                        Tensor &embeding_vector,
                                        const Tensor &embedding_vector_offset) {
  CudaDeviceContext context(core_->get_device_id());

  DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(num_key_per_table_offset_.dtype().type(), index_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();

      {
        constexpr int block_size = 256;
        int grid_size = (static_cast<int64_t>(num_keys) - 1) / block_size + 1;
        embedding_insert_kernel<<<grid_size, block_size, 0, stream>>>(
            keys.get<key_t>(), num_keys, num_unique_key_per_table_offset.get<uint32_t>(),
            num_table_offset, embeding_vector.get<float>(), embedding_vector_offset.get<uint32_t>(),
            table_id_list.get<int>(), table_ids_.get<int>(), table_ids_.get_num_elements(),
            keys_.get<key_t>(), num_key_per_table_offset_.get<index_t>(), emb_table_.get<float>(),
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

  *keys = keys_.to(core_, device);
  *id_space_offset = num_key_per_table_offset_.to(core_, device);
  *embedding_table = emb_table_.to(core_, device);
  *ev_size_list = local_ev_size_list_.to(core_, device);
  *id_space = table_ids_.to(core_, device);
}

void RaggedStaticEmbeddingTable::dump_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table,
                                            int table_id) {
  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = keys_.dtype();
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    key_t *d_keys = (key_t *)keys_.get();
    d_keys += h_num_key_per_table_offset_[table_index];
    key_t *h_keys = (key_t *)h_keys_tensor->get();
    HCTR_LIB_THROW(cudaMemcpy(h_keys, d_keys, sizeof(key_t) * h_num_key_per_table_[table_index],
                              cudaMemcpyDeviceToHost));

    float *d_embedding_vector = (float *)emb_table_.get();
    d_embedding_vector += h_emb_table_ev_offset_[table_index];
    float *h_embedding_vector = (float *)h_embedding_table->get();
    HCTR_LIB_THROW(cudaMemcpy(h_embedding_vector, d_embedding_vector,
                              sizeof(float) * h_size_per_table_[table_index],
                              cudaMemcpyDeviceToHost));
  });
}

void RaggedStaticEmbeddingTable::load_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table,
                                            int table_id) {
  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = keys_.dtype();

  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    DISPATCH_UNSIGNED_INTEGRAL_FUNCTION(num_key_per_table_offset_.dtype().type(), index_t, [&] {
      Tensor d_keys;
      Tensor d_embedding_vector;
      auto buffer_ptr = GetBuffer(core_);
      d_keys = buffer_ptr->reserve(h_keys_tensor->get_num_elements(), DeviceType::GPU, key_type);
      d_embedding_vector = buffer_ptr->reserve(h_embedding_table->get_num_elements(),
                                               DeviceType::GPU, TensorScalarType::Float32);
      buffer_ptr->allocate();

      d_keys.copy_from(*h_keys_tensor);
      d_embedding_vector.copy_from(*h_embedding_table);
      size_t max_vocabulary_size = h_table_max_vocabulary_size_[table_index];
      size_t num_keys = h_keys_tensor->get_num_elements();
      size_t table_keys = h_num_key_per_table_[table_index];

      {
        constexpr int block_size = 256;
        int grid_size =
            (static_cast<int64_t>(h_keys_tensor->get_num_elements()) - 1) / block_size + 1;
        embedding_insert_by_tableindex_kernel<<<grid_size, block_size>>>(
            (key_t *)d_keys.get(), num_keys, keys_.get<key_t>(),
            num_key_per_table_offset_.get<index_t>(), (float *)d_embedding_vector.get(),
            emb_table_.get<float>(), table_index, max_vocabulary_size,
            emb_table_ev_offset_.get<uint64_t>(), local_ev_size_list_.get<int>());
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
