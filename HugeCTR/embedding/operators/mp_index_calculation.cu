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
#include <cub/cub.cuh>

#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "mp_index_calculation.hpp"

namespace embedding {

namespace {

template <typename key_t, typename offset_t>
__global__ void index_calculation_kernel(const key_t* key, const offset_t* bucket_range,
                                         const int* local_embedding_list,
                                         const int* local_shard_id_list,
                                         const int* local_num_shards_list, int batch_size,
                                         int num_local_embedding, uint32_t* model_idx_offsets,
                                         char* flag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < batch_size * num_local_embedding) {
    int batch_id = tid % batch_size;
    int embedding_id = local_embedding_list[tid / batch_size];
    int shard_id = local_shard_id_list[tid / batch_size];
    int shards_count = local_num_shards_list[tid / batch_size];

    uint32_t bucket_start =
        static_cast<uint32_t>(bucket_range[batch_size * embedding_id + batch_id]);
    uint32_t bucket_end =
        static_cast<uint32_t>(bucket_range[batch_size * embedding_id + batch_id + 1]);
    uint32_t flag_cnt = 0;
    for (uint32_t idx = 0; idx < (bucket_end - bucket_start); ++idx) {
      key_t k = key[idx + bucket_start];
      if (k % shards_count == shard_id) {
        flag[idx + bucket_start] = 1;
        flag_cnt += 1;
      }
    }
    model_idx_offsets[1 + tid] = flag_cnt;
    if (tid == 0) {
      model_idx_offsets[0] = 0;
    }
  }
}

__global__ void expand_bucket_id_kernel(const uint32_t* model_offset, uint32_t* bucket_idx,
                                        int batch_size, int num_local_embedding,
                                        int batch_size_per_gpu) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < batch_size * num_local_embedding;
       idx += blockDim.x * gridDim.x) {
    uint32_t start = model_offset[idx];
    uint32_t end = model_offset[idx + 1];
    for (int i = start; i < end; ++i) {
      bucket_idx[i] = idx;
    }
  }
}

template <typename key_t>
constexpr key_t empty_key = std::numeric_limits<key_t>::max();

template <typename key_t>
class Hash {
 public:
  __forceinline__ __device__ Hash() {}
  __forceinline__ __device__ uint32_t operator()(key_t key) { return static_cast<uint32_t>(key); }
};

template <typename key_t>
__global__ void initialize_hash_key(key_t* hash_key, int num_hash_key) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_hash_key) {
    hash_key[idx] = empty_key<key_t>;
  }
}

__forceinline__ __device__ int32_t _atomicCAS(int32_t* address, int32_t compare, int32_t val) {
  return (int32_t)atomicCAS((int*)address, (int)compare, (int)val);
}

__forceinline__ __device__ uint32_t _atomicCAS(uint32_t* address, uint32_t compare, uint32_t val) {
  return (uint32_t)atomicCAS((unsigned int*)address, (unsigned int)compare, (unsigned int)val);
}

__forceinline__ __device__ int64_t _atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                            (unsigned long long)val);
}

__forceinline__ __device__ uint64_t _atomicCAS(uint64_t* address, uint64_t compare, uint64_t val) {
  return (uint64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                             (unsigned long long)val);
}

template <typename key_t, typename hasher_t = Hash<key_t>>
__global__ void get_unique_index_kernel(const key_t* key_list, size_t num_key,
                                        const uint32_t* id_space_offset, const int* id_space_list,
                                        size_t num_id_space, const int* unique_id_space_list,
                                        size_t num_unique_id_space, const uint32_t* hash_offset,
                                        key_t* hash_key_list, uint32_t* local_index) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  hasher_t hasher;

  if (idx < num_key) {
    uint32_t idx_id_space = binary_search_index_lower_bound(id_space_offset, num_id_space + 1, idx);
    int id_space = id_space_list[idx_id_space];
    int idx_unique_id_space =
        binary_search_index_lower_bound(unique_id_space_list, num_unique_id_space, id_space);

    uint32_t start = hash_offset[idx_unique_id_space];
    uint32_t end = hash_offset[idx_unique_id_space + 1];
    key_t target_key = key_list[idx];

    uint32_t capacity = end - start;
    uint32_t hash_index = hasher(target_key) % capacity;
    while (true) {
      const key_t old_key =
          _atomicCAS(hash_key_list + start + hash_index, (key_t)empty_key<key_t>, target_key);

      if ((empty_key<key_t> == old_key) || (target_key == old_key)) {
        local_index[idx] = start + hash_index;
        break;
      }
      hash_index = (hash_index + 1) % capacity;
    }
  }
}

__global__ void extract_wgrad_dst_idx_kernel(const uint32_t* unique_local_index,
                                             size_t num_unique_key, uint32_t* wgrad_dst_idx) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_unique_key) {
    uint32_t local_index = unique_local_index[idx];
    wgrad_dst_idx[idx] = (idx > 0 && unique_local_index[idx - 1] != local_index) ? 1 : 0;
  }
}

__global__ void extract_wgrad_ev_dst_idx_kernel(const uint32_t* hash_offset,
                                                int num_unique_id_space_list,
                                                const uint32_t* unique_local_index,
                                                size_t num_unique_key, const int* ev_size_per_table,
                                                uint32_t* wgrad_dst_idx) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_unique_key) {
    uint32_t local_index = unique_local_index[idx];

    uint32_t idx_id_space =
        binary_search_index_lower_bound(hash_offset, num_unique_id_space_list + 1, local_index);

    wgrad_dst_idx[1 + idx] = (idx == 0 || unique_local_index[idx - 1] != local_index)
                                 ? ev_size_per_table[idx_id_space]
                                 : 0;
  }

  if (idx == 0) {
    wgrad_dst_idx[0] = 0;
  }
}

template <typename key_t>
__global__ void convert_hash_index_to_key_kernel(const uint32_t* hash_index, size_t num_hash_index,
                                                 const key_t* hash_keys, key_t* key) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_hash_index;
       tid += blockDim.x * gridDim.x) {
    uint32_t index = hash_index[tid];
    key[tid] = hash_keys[index];
  }
}

template <typename key_t, int kWarpPerBlock = 1, int kWarpSize = 32>
__global__ void count_unique_key_kernel(const key_t* hash_keys, const uint32_t* hash_offset,
                                        int num_unique_id_space, uint32_t* unique_key_count) {
  int warp_id = 0;
  int lane_id = threadIdx.x;
  int block_id = blockIdx.x;

  int count = 0;
  if (block_id < num_unique_id_space) {
    int start = hash_offset[block_id];
    int end = hash_offset[block_id + 1];
    for (int i = 0; i * kWarpSize + lane_id < (end - start); ++i) {
      count += (hash_keys[start + i * kWarpSize + lane_id] == empty_key<key_t>) ? 0 : 1;
    }
  }

  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[kWarpPerBlock];
  int aggregate = WarpReduce(temp_storage[warp_id]).Sum(count);

  if (lane_id == 0) {
    unique_key_count[block_id + 1] = aggregate;
    if (block_id == 0) {
      unique_key_count[0] = 0;
    }
  }
}

template <typename key_t, int kWarpPerBlock, int kWarpSize = 32>
__global__ void scan_id_space_offset(const key_t* hash_keys, const uint32_t* hash_offset,
                                     int num_unique_id_space, uint32_t* unique_id_space_offset,
                                     uint32_t* temp_id_space_value) {
  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;

  int count = 0;
  if (warp_id < num_unique_id_space) {
    int start = hash_offset[warp_id];
    int end = hash_offset[warp_id + 1];
    for (int i = 0; i * kWarpSize + lane_id < (end - start); ++i) {
      count += (hash_keys[start + i * kWarpSize + lane_id] == empty_key<key_t>) ? 0 : 1;
    }
  }

  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[kWarpPerBlock];
  int aggregate = WarpReduce(temp_storage[warp_id]).Sum(count);

  __shared__ int s_id_space_offset[kWarpPerBlock];
  if (lane_id == 0) {
    s_id_space_offset[warp_id] = aggregate;
  }
  __syncthreads();

  if (threadIdx.x + threadIdx.y * blockDim.x == 0) {
    uint32_t prefix_sum = 0;
    for (int i = 0; i < num_unique_id_space + 1; ++i) {
      unique_id_space_offset[i] = prefix_sum;

      prefix_sum += static_cast<uint32_t>(s_id_space_offset[i]);
    }
  }
}
}  // namespace

ModelIndexCalculation::ModelIndexCalculation(std::shared_ptr<CoreResourceManager> core,
                                             int num_local_embedding, int local_hotness_sum,
                                             int hotness_sum, int universal_batch_size,
                                             DataType key_type)
    : core_(core),
      num_local_embedding_(num_local_embedding),
      local_hotness_sum_(local_hotness_sum),
      hotness_list_sum_(hotness_sum),
      universal_batch_size_(universal_batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  Device device{DeviceType::GPU, core->get_device_id()};

  auto buffer_ptr_ = GetBuffer(core);
  model_key_ = buffer_ptr_->reserve({universal_batch_size_ * local_hotness_sum_}, device, key_type);
  model_idx_offsets_ = buffer_ptr_->reserve({universal_batch_size_ * num_local_embedding_ + 1},
                                            device, TensorScalarType::UInt32);
  model_sp_weight_ = buffer_ptr_->reserve({universal_batch_size_ * local_hotness_sum_}, device,
                                          TensorScalarType::Float32);
  num_key_in_bucket_for_combiner_ = buffer_ptr_->reserve(
      {universal_batch_size_ * num_local_embedding_}, device, TensorScalarType::UInt32);
  num_model_key_ = buffer_ptr_->reserve({1}, DeviceType::CPU, TensorScalarType::Size_t);
  flag_ = buffer_ptr_->reserve({universal_batch_size_ * hotness_list_sum_}, device,
                               TensorScalarType::Char);
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  universal_batch_size_ * num_local_embedding_ + 1);
    d_temp_scan_storage_ = buffer_ptr_->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
      cub::DeviceSelect::Flagged(nullptr, temp_bytes, (key_t*)nullptr, (char*)nullptr,
                                 (key_t*)nullptr, (size_t*)nullptr,
                                 universal_batch_size * hotness_list_sum_);
    });
    d_temp_select_storage_ = buffer_ptr_->reserve({temp_bytes}, device, TensorScalarType::Void);
  }

  size_t temp_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, temp_bytes, (size_t*)nullptr, (char*)nullptr,
                             (size_t*)nullptr, (size_t*)nullptr,
                             universal_batch_size * hotness_list_sum_);
  d_temp_select_weight_storage_ =
      buffer_ptr_->reserve({temp_bytes}, device, TensorScalarType::Void);
  buffer_ptr_->allocate();
}

void ModelIndexCalculation::compute(const Tensor& key, const Tensor& bucket_range, size_t num_key,
                                    const Tensor& d_local_embedding_list,
                                    const Tensor& d_local_shard_id_list,
                                    const Tensor& d_local_num_shards_list, int batch_size,
                                    Tensor* model_key, Tensor* model_idx_offsets,
                                    size_t* num_model_key) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  *(num_model_key_.get<size_t>()) = 0;
  if (num_local_embedding_ > 0) {
    DISPATCH_INTEGRAL_FUNCTION(key.dtype().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();

        HCTR_LIB_THROW(cudaMemsetAsync(model_key_.get(), 0, model_key_.nbytes(), stream));
        HCTR_LIB_THROW(
            cudaMemsetAsync(model_idx_offsets_.get(), 0, model_idx_offsets_.nbytes(), stream));
        HCTR_LIB_THROW(cudaMemsetAsync(flag_.get(), 0, flag_.nbytes(), stream));

        key_t* model_key_ptr = model_key_.get<key_t>();
        uint32_t* model_idx_offsets_ptr = model_idx_offsets_.get<uint32_t>();
        size_t* num_model_key_ptr = num_model_key_.get<size_t>();
        char* flag_ptr = flag_.get<char>();
        const key_t* key_ptr = key.get<key_t>();
        const offset_t* bucket_range_ptr = bucket_range.get<offset_t>();
        const int* local_embedding_list_ptr = d_local_embedding_list.get<int>();
        const int* local_shard_id_ptr = d_local_shard_id_list.get<int>();
        const int* local_num_shards_ptr = d_local_num_shards_list.get<int>();

        // in cub implementation, the flag must be 0 or 1. See
        // https://github.com/NVIDIA/cub/issues/235 we can fuse thie memset with next kernel
        int thread_cnt = 128;
        int block_cnt = (batch_size * num_local_embedding_ - 1) / thread_cnt + 1;
        index_calculation_kernel<<<block_cnt, thread_cnt, 0, stream>>>(
            key_ptr, bucket_range_ptr, local_embedding_list_ptr, local_shard_id_ptr,
            local_num_shards_ptr, batch_size, num_local_embedding_, model_idx_offsets_ptr,
            flag_ptr);

        size_t d_temp_scan_storage_nbytes = d_temp_scan_storage_.nbytes();
        cub::DeviceScan::InclusiveSum(d_temp_scan_storage_.get(), d_temp_scan_storage_nbytes,
                                      model_idx_offsets_ptr, model_idx_offsets_ptr,
                                      batch_size * num_local_embedding_ + 1, stream);

        size_t d_temp_select_storage_nbytes = d_temp_select_storage_.nbytes();
        cub::DeviceSelect::Flagged(d_temp_select_storage_.get(), d_temp_select_storage_nbytes,
                                   key_ptr, flag_ptr, model_key_ptr, num_model_key_ptr, num_key,
                                   stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      });
    });
  }
  *model_key = model_key_;
  *model_idx_offsets = model_idx_offsets_;
  *num_model_key = *(num_model_key_.get<size_t>());
}
void ModelIndexCalculation::compute(const Tensor& key, const Tensor& bucket_range, size_t num_key,
                                    const Tensor& d_local_embedding_list,
                                    const Tensor& d_local_shard_id_list,
                                    const Tensor& d_local_num_shards_list, int batch_size,
                                    Tensor* model_key, Tensor* model_idx_offsets,
                                    size_t* num_model_key, const Tensor& reorder_sp_weight,
                                    Tensor* model_sp_weight) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  *(num_model_key_.get<size_t>()) = 0;
  if (num_local_embedding_ > 0) {
    DISPATCH_INTEGRAL_FUNCTION(key.dtype().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();

        HCTR_LIB_THROW(cudaMemsetAsync(model_key_.get(), 0, model_key_.nbytes(), stream));
        HCTR_LIB_THROW(
            cudaMemsetAsync(model_idx_offsets_.get(), 0, model_idx_offsets_.nbytes(), stream));
        HCTR_LIB_THROW(cudaMemsetAsync(flag_.get(), 0, flag_.nbytes(), stream));

        key_t* model_key_ptr = model_key_.get<key_t>();
        uint32_t* model_idx_offsets_ptr = model_idx_offsets_.get<uint32_t>();
        size_t* num_model_key_ptr = num_model_key_.get<size_t>();
        char* flag_ptr = flag_.get<char>();
        const key_t* key_ptr = key.get<key_t>();
        const offset_t* bucket_range_ptr = bucket_range.get<offset_t>();
        const int* local_embedding_list_ptr = d_local_embedding_list.get<int>();
        const int* local_shard_id_ptr = d_local_shard_id_list.get<int>();
        const int* local_num_shards_ptr = d_local_num_shards_list.get<int>();

        // in cub implementation, the flag must be 0 or 1. See
        // https://github.com/NVIDIA/cub/issues/235 we can fuse thie memset with next kernel
        int thread_cnt = 128;
        int block_cnt = (batch_size * num_local_embedding_ - 1) / thread_cnt + 1;
        index_calculation_kernel<<<block_cnt, thread_cnt, 0, stream>>>(
            key_ptr, bucket_range_ptr, local_embedding_list_ptr, local_shard_id_ptr,
            local_num_shards_ptr, batch_size, num_local_embedding_, model_idx_offsets_ptr,
            flag_ptr);

        size_t d_temp_scan_storage_nbytes = d_temp_scan_storage_.nbytes();
        cub::DeviceScan::InclusiveSum(d_temp_scan_storage_.get(), d_temp_scan_storage_nbytes,
                                      model_idx_offsets_ptr, model_idx_offsets_ptr,
                                      batch_size * num_local_embedding_ + 1, stream);

        size_t d_temp_select_storage_nbytes = d_temp_select_storage_.nbytes();
        cub::DeviceSelect::Flagged(d_temp_select_storage_.get(), d_temp_select_storage_nbytes,
                                   key_ptr, flag_ptr, model_key_ptr, num_model_key_ptr, num_key,
                                   stream);
        DISPATCH_FLOAT_AND_HALF_FUNCTION(reorder_sp_weight.dtype().type(), dtype_t, [&] {
          const dtype_t* reorder_sp_weight_ptr = reorder_sp_weight.get<dtype_t>();
          dtype_t* model_sp_weight_ptr = model_sp_weight_.get<dtype_t>();
          size_t d_temp_select_weight_storage_nbytes = d_temp_select_weight_storage_.nbytes();

          cub::DeviceSelect::Flagged(d_temp_select_weight_storage_.get(),
                                     d_temp_select_weight_storage_nbytes, reorder_sp_weight_ptr,
                                     flag_ptr, model_sp_weight_ptr, num_model_key_ptr, num_key,
                                     stream);
        });
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      });
    });
  }
  *model_sp_weight = model_sp_weight_;
  *model_key = model_key_;
  *model_idx_offsets = model_idx_offsets_;
  *num_model_key = *(num_model_key_.get<size_t>());
}

ModelBackwardIndexCalculation::ModelBackwardIndexCalculation(
    std::shared_ptr<CoreResourceManager> core, int num_gpus, int num_local_embedding,
    const std::vector<int>& h_local_hotness_list, const std::vector<int>& h_local_id_space_list,
    const std::vector<int>& h_local_ev_size_list, int universal_batch_size, DataType key_type)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(num_local_embedding) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  Device device{DeviceType::GPU};

  int local_hotness_sum =
      std::accumulate(h_local_hotness_list.begin(), h_local_hotness_list.end(), 0);
  sort_end_bit_ =
      static_cast<int>(log2(static_cast<float>(universal_batch_size * local_hotness_sum))) + 1;

  std::vector<int> h_unique_id_space_list;
  std::vector<int> h_unique_id_space_ev_size_list;
  for (size_t i = 0; i < h_local_id_space_list.size(); ++i) {
    if (h_unique_id_space_list.size() == 0) {
      h_unique_id_space_list.push_back(h_local_id_space_list[i]);
      h_unique_id_space_ev_size_list.push_back(h_local_ev_size_list[i]);
      continue;
    }
    if (h_local_id_space_list[i] > h_unique_id_space_list.back()) {
      h_unique_id_space_list.push_back(h_local_id_space_list[i]);
      h_unique_id_space_ev_size_list.push_back(h_local_ev_size_list[i]);
    }
  }

  auto buffer_ptr = GetBuffer(core);
  bucket_id_list_ = buffer_ptr->reserve(universal_batch_size * local_hotness_sum, DeviceType::GPU,
                                        TensorScalarType::UInt32);
  hash_keys_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device, key_type);
  hash_offset_ =
      buffer_ptr->reserve({1 + h_unique_id_space_list.size()}, device, TensorScalarType::UInt32);
  local_index_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                     TensorScalarType::UInt32);
  sorted_local_index_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                            TensorScalarType::UInt32);
  unique_local_index_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                            TensorScalarType::UInt32);

  unique_key_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device, key_type);
  num_unique_key_ = buffer_ptr->reserve({1}, DeviceType::CPU, TensorScalarType::Size_t);
  unique_dst_idx_ = buffer_ptr->reserve({1 + universal_batch_size * local_hotness_sum}, device,
                                        TensorScalarType::UInt32);
  sorted_bucket_id_list_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                               TensorScalarType::UInt32);

  sorted_bucket_id_offset_ = buffer_ptr->reserve({1 + universal_batch_size * local_hotness_sum},
                                                 device, TensorScalarType::UInt32);
  unique_id_space_offset_ =
      buffer_ptr->reserve({1 + h_unique_id_space_list.size()}, device, TensorScalarType::UInt32);
  unique_id_space_list_ = buffer_ptr->reserve({h_unique_id_space_list.size()}, DeviceType::GPU,
                                              TensorScalarType::Int32);
  unique_id_space_ev_size_list_ = buffer_ptr->reserve({h_unique_id_space_ev_size_list.size()},
                                                      DeviceType::GPU, TensorScalarType::Int32);
  coordinate_wgrad_dst_idx_ = buffer_ptr->reserve({1 + universal_batch_size * local_hotness_sum},
                                                  device, TensorScalarType::UInt32);
  {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                    (uint32_t*)nullptr, (uint32_t*)nullptr,
                                    universal_batch_size * local_hotness_sum, 0, sort_end_bit_);
    d_temp_sort_storage_ = buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }

  {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, (size_t*)nullptr, (size_t*)nullptr,
                                    (size_t*)nullptr, (size_t*)nullptr,
                                    universal_batch_size * local_hotness_sum, 0, sort_end_bit_);
    d_temp_sort_sp_weight_storage_ =
        buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
    d_temp_sort_sp_weight_key_ = buffer_ptr->reserve({universal_batch_size * local_hotness_sum},
                                                     device, TensorScalarType::UInt32);
    sorted_sp_weight_list_ = buffer_ptr->reserve({universal_batch_size * local_hotness_sum}, device,
                                                 TensorScalarType::Float32);
  }

  {
    size_t temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                       (uint32_t*)nullptr, (size_t*)nullptr,
                                       universal_batch_size * local_hotness_sum);
    d_temp_run_length_encode_storage_ =
        buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
        std::max(static_cast<int64_t>(1 + universal_batch_size * local_hotness_sum),
                 unique_id_space_offset_.get_num_elements()));
    d_temp_scan_encode_storage_ = buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  buffer_ptr->allocate();
  unique_id_space_list_.copy_from(h_unique_id_space_list);
  unique_id_space_ev_size_list_.copy_from(h_unique_id_space_ev_size_list);

  std::vector<uint32_t> h_hash_offset(1 + h_unique_id_space_list.size(), 0);
  for (int i = 0; i < num_local_embedding; ++i) {
    int id_space = h_local_id_space_list[i];
    auto iter = find(h_unique_id_space_list.begin(), h_unique_id_space_list.end(), id_space);
    HCTR_CHECK_HINT(iter != h_unique_id_space_list.end(),
                    "can not find id space in unique id space");
    int idx = std::distance(h_unique_id_space_list.begin(), iter);
    h_hash_offset[1 + idx] += universal_batch_size * h_local_hotness_list[i];
  }
  std::partial_sum(h_hash_offset.begin(), h_hash_offset.end(), h_hash_offset.begin());
  hash_offset_.copy_from(h_hash_offset);
}

void ModelBackwardIndexCalculation::compute(
    const Tensor& model_key, size_t num_model_key, const Tensor& model_offset,
    const Tensor& id_space_offset, const Tensor& id_space_list, int batch_size, Tensor* unique_key,
    size_t* num_unique_key, Tensor* unique_dst_idx, Tensor* sorted_bucket_id_list,
    Tensor* sorted_bucket_id_offset, Tensor* unique_id_space_list, Tensor* unique_id_space_offset,
    Tensor* coordinate_key, Tensor* coordinate_wgrad_dst_idx, const Tensor& model_sp_weight,
    Tensor* coordinate_sp_weight) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_INTEGRAL_FUNCTION(model_key.dtype().type(), key_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();

    HCTR_LIB_THROW(
        cudaMemsetAsync(bucket_id_list_.get<uint32_t>(), 0, bucket_id_list_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_local_index_.get<uint32_t>(), 0,
                                   sorted_local_index_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(unique_local_index_.get<uint32_t>(), 0,
                                   unique_local_index_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(unique_key_.get<key_t>(), 0, unique_key_.nbytes(), stream));
    HCTR_LIB_THROW(
        cudaMemsetAsync(unique_dst_idx_.get<uint32_t>(), 0, unique_dst_idx_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(coordinate_wgrad_dst_idx_.get<uint32_t>(), 0,
                                   coordinate_wgrad_dst_idx_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_list_.get<uint32_t>(), 0,
                                   sorted_bucket_id_list_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_offset_.get<uint32_t>(), 0,
                                   sorted_bucket_id_offset_.nbytes(), stream));
    // TODO:: need to fix  a flexsible type
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_sp_weight_list_.get<float>(), 0,
                                   sorted_sp_weight_list_.nbytes(), stream));
    if (num_local_embedding_ > 0 && num_model_key > 0ul) {
      {
        // this can be fused with sort pair in 4th code
        int block_size = 256;
        int grid_size = (batch_size * num_local_embedding_ - 1) / block_size + 1;
        expand_bucket_id_kernel<<<grid_size, block_size, 0, stream>>>(
            model_offset.get<uint32_t>(), bucket_id_list_.get<uint32_t>(), batch_size,
            num_local_embedding_, batch_size_per_gpu);
      }
      {
        int num_hash_key = hash_keys_.get_num_elements();
        constexpr int block_size = 256;
        int grid_size = (num_hash_key - 1) / block_size + 1;
        initialize_hash_key<<<grid_size, block_size, 0, stream>>>(hash_keys_.get<key_t>(),
                                                                  num_hash_key);
      }
      {
        constexpr int block_size = 256;
        int grid_size = (num_model_key - 1) / block_size + 1;
        get_unique_index_kernel<<<grid_size, block_size, 0, stream>>>(
            model_key.get<key_t>(), num_model_key, id_space_offset.get<uint32_t>(),
            id_space_list.get<int>(), num_local_embedding_, unique_id_space_list_.get<int>(),
            unique_id_space_list_.get_num_elements(), hash_offset_.get<uint32_t>(),
            hash_keys_.get<key_t>(), local_index_.get<uint32_t>());
      }
      {
        DISPATCH_FLOAT_AND_HALF_FUNCTION(model_sp_weight.dtype().type(), dtype_t, [&] {
          size_t nbytes = d_temp_sort_sp_weight_storage_.nbytes();
          cub::DeviceRadixSort::SortPairs(
              d_temp_sort_sp_weight_storage_.get(), nbytes, local_index_.get<uint32_t>(),
              d_temp_sort_sp_weight_key_.get<uint32_t>(), model_sp_weight.get<dtype_t>(),
              sorted_sp_weight_list_.get<dtype_t>(), num_model_key, 0, sort_end_bit_, stream);
        });
      }
      {
        size_t nbytes = d_temp_sort_storage_.nbytes();
        cub::DeviceRadixSort::SortPairs(
            d_temp_sort_storage_.get(), nbytes, local_index_.get<uint32_t>(),
            sorted_local_index_.get<uint32_t>(), bucket_id_list_.get<uint32_t>(),
            sorted_bucket_id_list_.get<uint32_t>(), num_model_key, 0, sort_end_bit_, stream);
      }
      {
        size_t nbytes = d_temp_run_length_encode_storage_.nbytes();
        cub::DeviceRunLengthEncode::Encode(
            d_temp_run_length_encode_storage_.get(), nbytes, sorted_local_index_.get<uint32_t>(),
            unique_local_index_.get<uint32_t>(), sorted_bucket_id_offset_.get<uint32_t>() + 1,
            num_unique_key_.get<size_t>(), num_model_key, stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));  // to sync num_unique_key to host
      }
      int num_unique_table = unique_id_space_list_.get_num_elements();
      {
        constexpr int block_size = 256;
        int grid_size = (num_model_key - 1) / block_size + 1;
        extract_wgrad_dst_idx_kernel<<<grid_size, block_size, 0, stream>>>(
            sorted_local_index_.get<uint32_t>(), num_model_key,
            coordinate_wgrad_dst_idx_.get<uint32_t>());
      }
      {
        constexpr int block_size = 256;
        int num_unique_key_host = *num_unique_key_.get<size_t>();
        int grid_size = (num_unique_key_host - 1) / block_size + 1;
        convert_hash_index_to_key_kernel<<<grid_size, block_size, 0, stream>>>(
            unique_local_index_.get<uint32_t>(), num_unique_key_host, hash_keys_.get<key_t>(),
            unique_key_.get<key_t>());
        extract_wgrad_ev_dst_idx_kernel<<<grid_size, block_size, 0, stream>>>(
            hash_offset_.get<uint32_t>(), num_unique_table, unique_local_index_.get<uint32_t>(),
            num_unique_key_host, unique_id_space_ev_size_list_.get<int>(),
            unique_dst_idx_.get<uint32_t>());
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      {
        int num_unique_id_space = static_cast<int>(unique_id_space_list_.get_num_elements());
        count_unique_key_kernel<<<num_unique_id_space, 32, 0, stream>>>(
            hash_keys_.get<key_t>(), hash_offset_.get<uint32_t>(), num_unique_id_space,
            unique_id_space_offset_.get<uint32_t>());

        HCTR_LIB_THROW(cudaPeekAtLastError());
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      {
        size_t nbytes = d_temp_scan_encode_storage_.nbytes();
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      unique_id_space_offset_.get<uint32_t>(),
                                      unique_id_space_offset_.get<uint32_t>(),
                                      unique_id_space_offset_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(
            d_temp_scan_encode_storage_.get(), nbytes, unique_dst_idx_.get<uint32_t>(),
            unique_dst_idx_.get<uint32_t>(), unique_dst_idx_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      coordinate_wgrad_dst_idx_.get<uint32_t>(),
                                      coordinate_wgrad_dst_idx_.get<uint32_t>(),
                                      coordinate_wgrad_dst_idx_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get_num_elements(), stream);
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
  });
  *unique_key = unique_key_;
  *num_unique_key = *num_unique_key_.get<size_t>();
  *unique_dst_idx = unique_dst_idx_;
  *sorted_bucket_id_list = sorted_bucket_id_list_;
  *sorted_bucket_id_offset = sorted_bucket_id_offset_;
  *unique_id_space_list = unique_id_space_list_;
  *unique_id_space_offset = unique_id_space_offset_;
  *coordinate_key = sorted_local_index_;
  *coordinate_wgrad_dst_idx = coordinate_wgrad_dst_idx_;
  *coordinate_sp_weight = sorted_sp_weight_list_;
}

void ModelBackwardIndexCalculation::compute(
    const Tensor& model_key, size_t num_model_key, const Tensor& model_offset,
    const Tensor& id_space_offset, const Tensor& id_space_list, int batch_size, Tensor* unique_key,
    size_t* num_unique_key, Tensor* unique_dst_idx, Tensor* sorted_bucket_id_list,
    Tensor* sorted_bucket_id_offset, Tensor* unique_id_space_list, Tensor* unique_id_space_offset,
    Tensor* coordinate_key, Tensor* coordinate_wgrad_dst_idx) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;

  DISPATCH_INTEGRAL_FUNCTION(model_key.dtype().type(), key_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();

    HCTR_LIB_THROW(
        cudaMemsetAsync(bucket_id_list_.get<uint32_t>(), 0, bucket_id_list_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_local_index_.get<uint32_t>(), 0,
                                   sorted_local_index_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(unique_local_index_.get<uint32_t>(), 0,
                                   unique_local_index_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(unique_key_.get<key_t>(), 0, unique_key_.nbytes(), stream));
    HCTR_LIB_THROW(
        cudaMemsetAsync(unique_dst_idx_.get<uint32_t>(), 0, unique_dst_idx_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(coordinate_wgrad_dst_idx_.get<uint32_t>(), 0,
                                   coordinate_wgrad_dst_idx_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_list_.get<uint32_t>(), 0,
                                   sorted_bucket_id_list_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_offset_.get<uint32_t>(), 0,
                                   sorted_bucket_id_offset_.nbytes(), stream));
    if (num_local_embedding_ > 0 && num_model_key > 0ul) {
      {
        // this can be fused with sort pair in 4th code
        int block_size = 256;
        int grid_size = (batch_size * num_local_embedding_ - 1) / block_size + 1;
        expand_bucket_id_kernel<<<grid_size, block_size, 0, stream>>>(
            model_offset.get<uint32_t>(), bucket_id_list_.get<uint32_t>(), batch_size,
            num_local_embedding_, batch_size_per_gpu);
      }
      {
        int num_hash_key = hash_keys_.get_num_elements();
        constexpr int block_size = 256;
        int grid_size = (num_hash_key - 1) / block_size + 1;
        initialize_hash_key<<<grid_size, block_size, 0, stream>>>(hash_keys_.get<key_t>(),
                                                                  num_hash_key);
      }
      {
        constexpr int block_size = 256;
        int grid_size = (num_model_key - 1) / block_size + 1;
        get_unique_index_kernel<<<grid_size, block_size, 0, stream>>>(
            model_key.get<key_t>(), num_model_key, id_space_offset.get<uint32_t>(),
            id_space_list.get<int>(), num_local_embedding_, unique_id_space_list_.get<int>(),
            unique_id_space_list_.get_num_elements(), hash_offset_.get<uint32_t>(),
            hash_keys_.get<key_t>(), local_index_.get<uint32_t>());
      }
      {
        size_t nbytes = d_temp_sort_storage_.nbytes();
        cub::DeviceRadixSort::SortPairs(
            d_temp_sort_storage_.get(), nbytes, local_index_.get<uint32_t>(),
            sorted_local_index_.get<uint32_t>(), bucket_id_list_.get<uint32_t>(),
            sorted_bucket_id_list_.get<uint32_t>(), num_model_key, 0, sort_end_bit_, stream);
      }
      {
        size_t nbytes = d_temp_run_length_encode_storage_.nbytes();
        cub::DeviceRunLengthEncode::Encode(
            d_temp_run_length_encode_storage_.get(), nbytes, sorted_local_index_.get<uint32_t>(),
            unique_local_index_.get<uint32_t>(), sorted_bucket_id_offset_.get<uint32_t>() + 1,
            num_unique_key_.get<size_t>(), num_model_key, stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));  // to sync num_unique_key to host
      }
      int num_unique_table = unique_id_space_list_.get_num_elements();
      {
        constexpr int block_size = 256;
        int grid_size = (num_model_key - 1) / block_size + 1;
        extract_wgrad_dst_idx_kernel<<<grid_size, block_size, 0, stream>>>(
            sorted_local_index_.get<uint32_t>(), num_model_key,
            coordinate_wgrad_dst_idx_.get<uint32_t>());
      }
      {
        constexpr int block_size = 256;
        int num_unique_key_host = *num_unique_key_.get<size_t>();
        int grid_size = (num_unique_key_host - 1) / block_size + 1;
        convert_hash_index_to_key_kernel<<<grid_size, block_size, 0, stream>>>(
            unique_local_index_.get<uint32_t>(), num_unique_key_host, hash_keys_.get<key_t>(),
            unique_key_.get<key_t>());
        extract_wgrad_ev_dst_idx_kernel<<<grid_size, block_size, 0, stream>>>(
            hash_offset_.get<uint32_t>(), num_unique_table, unique_local_index_.get<uint32_t>(),
            num_unique_key_host, unique_id_space_ev_size_list_.get<int>(),
            unique_dst_idx_.get<uint32_t>());
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      {
        int num_unique_id_space = static_cast<int>(unique_id_space_list_.get_num_elements());
        count_unique_key_kernel<<<num_unique_id_space, 32, 0, stream>>>(
            hash_keys_.get<key_t>(), hash_offset_.get<uint32_t>(), num_unique_id_space,
            unique_id_space_offset_.get<uint32_t>());

        HCTR_LIB_THROW(cudaPeekAtLastError());
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      {
        size_t nbytes = d_temp_scan_encode_storage_.nbytes();
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      unique_id_space_offset_.get<uint32_t>(),
                                      unique_id_space_offset_.get<uint32_t>(),
                                      unique_id_space_offset_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(
            d_temp_scan_encode_storage_.get(), nbytes, unique_dst_idx_.get<uint32_t>(),
            unique_dst_idx_.get<uint32_t>(), unique_dst_idx_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      coordinate_wgrad_dst_idx_.get<uint32_t>(),
                                      coordinate_wgrad_dst_idx_.get<uint32_t>(),
                                      coordinate_wgrad_dst_idx_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(d_temp_scan_encode_storage_.get(), nbytes,
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get_num_elements(), stream);
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
  });
  *unique_key = unique_key_;
  *num_unique_key = *num_unique_key_.get<size_t>();
  *unique_dst_idx = unique_dst_idx_;
  *sorted_bucket_id_list = sorted_bucket_id_list_;
  *sorted_bucket_id_offset = sorted_bucket_id_offset_;
  *unique_id_space_list = unique_id_space_list_;
  *unique_id_space_offset = unique_id_space_offset_;
  *coordinate_key = sorted_local_index_;
  *coordinate_wgrad_dst_idx = coordinate_wgrad_dst_idx_;
}
}  // namespace embedding
