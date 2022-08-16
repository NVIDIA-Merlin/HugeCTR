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
#include "dp_index_calculation.hpp"
#include "generic_lookup.cuh"
namespace embedding {

namespace {

template <typename offset_t>
__global__ void mask_flag_kernel(int num_local_embedding, int batch_size_per_gpu,
                                 int const* d_local_embedding_list, int batch_size, int gpu_id,
                                 offset_t const* bucket_range, char* flag, uint32_t* dp_offset,
                                 uint32_t* dp_dst) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_local_embedding * batch_size_per_gpu) {
    int batch_id = tid % batch_size_per_gpu;
    int embedding_id = d_local_embedding_list[tid / batch_size_per_gpu];
    int bucket_id = batch_size * embedding_id + batch_size_per_gpu * gpu_id + batch_id;

    int start = bucket_range[bucket_id];
    int end = bucket_range[bucket_id + 1];
    dp_offset[1 + tid] = end - start;
    for (int j = start; j < end; ++j) {
      flag[j] = 1;
    }

    int dst_bucket_id = batch_size_per_gpu * embedding_id + batch_id;
    dp_dst[tid] = dst_bucket_id;
  }
}

template <typename key_t, typename offset_t>
__global__ void fused_select_dp_key_and_bucket_id_kernel(
    const key_t* keys, const offset_t* bucket_range, size_t num_key, int batch_size,
    int num_local_embedding, int num_embedding, int gpu_id, int num_gpu, const int* id_space_list,
    const int* local_embedding_list, key_t* dp_keys, uint32_t* dp_bucket_id,
    const int* segment_start_offsets, int* segment_end_offsets) {
  int local_embedding_id = blockIdx.x;
  int embedding_id = local_embedding_list[local_embedding_id];
  // int id_space = id_space_list[local_embedding_id];
  int batch_size_per_gpu = batch_size / num_gpu;

  uint32_t segment_start = segment_start_offsets[local_embedding_id];
  offset_t bucket_start = bucket_range[batch_size * embedding_id];
  for (int batch_id = threadIdx.x; batch_id < batch_size; batch_id += blockDim.x) {
    uint32_t bucket_id = batch_size * embedding_id + batch_id;

    uint32_t start = bucket_range[bucket_id];
    uint32_t end = bucket_range[bucket_id + 1];

    uint32_t local_bucket_id;
    if (batch_id >= gpu_id * batch_size_per_gpu && batch_id < (gpu_id + 1) * batch_size_per_gpu) {
      local_bucket_id = batch_size_per_gpu * embedding_id + batch_id % batch_size_per_gpu;
    } else {
      local_bucket_id = batch_size * num_embedding;
    }

    for (uint32_t r = start; r < end; ++r) {
      dp_keys[segment_start + (r - bucket_start)] = keys[r];
      dp_bucket_id[segment_start + (r - bucket_start)] = local_bucket_id;
    }
  }
  if (threadIdx.x == 0) {
    offset_t bucket_end = bucket_range[batch_size * embedding_id + batch_size];
    int num_key_in_bucket = static_cast<int>(bucket_end) - static_cast<int>(bucket_start);
    segment_end_offsets[local_embedding_id] = num_key_in_bucket + static_cast<int>(segment_start);
  }
}

template <typename key_t>
class SelectUniqueDPKeyOp {
  const key_t* sorted_dp_keys_;
  const int* segment_start_offsets_;
  const int* segment_end_offsets_;
  int num_embedding_;

 public:
  __host__ __device__ __forceinline__ SelectUniqueDPKeyOp(const key_t* sorted_dp_keys,
                                                          const int* segment_start_offsets,
                                                          const int* segment_end_offsets,
                                                          int num_embedding)
      : sorted_dp_keys_(sorted_dp_keys),
        segment_start_offsets_(segment_start_offsets),
        segment_end_offsets_(segment_end_offsets),
        num_embedding_(num_embedding) {}

  __device__ __forceinline__ bool operator()(const uint32_t& idx) const {
    int embedding_id = binary_search_index_lower_bound(segment_start_offsets_, num_embedding_ + 1,
                                                       static_cast<int>(idx));
    if (idx >= segment_end_offsets_[embedding_id]) return false;
    if (idx == segment_start_offsets_[embedding_id]) return true;
    return sorted_dp_keys_[idx] != sorted_dp_keys_[idx - 1];
  }
};

class SelectLocalBucketidOP {
  int empty_bucket_id_;

 public:
  __host__ __device__ __forceinline__ SelectLocalBucketidOP(int empty_bucket_id)
      : empty_bucket_id_(empty_bucket_id) {}

  __device__ __forceinline__ bool operator()(const uint32_t& bucket_id) const {
    return bucket_id < empty_bucket_id_;
  }
};

template <typename key_t>
__global__ void fused_compact_unique_key_and_count_bucket_id_offset(
    const key_t* key, const uint32_t* indices, const size_t* num_indices,
    const uint32_t* sorted_dp_bucket_id_list, const int* segment_start_offsets,
    const int* segment_end_offsets, const int* local_ev_size_list, int batch_size,
    int num_local_embedding, int num_embedding, key_t* compact_key, uint32_t* dst_idx,
    uint32_t* bucket_offset, uint32_t* unique_id_space_offset) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < *num_indices) {
    int idx = indices[tid];
    compact_key[tid] = key[idx];
    int embedding_id =
        binary_search_index_lower_bound(segment_start_offsets, num_local_embedding + 1, idx);
    if (segment_start_offsets[embedding_id] == idx) {
      unique_id_space_offset[embedding_id] = tid;
    }

    int ev_size = local_ev_size_list[embedding_id];
    dst_idx[tid + 1] = ev_size;
    int next_idx =
        (tid == *num_indices - 1) ? segment_start_offsets[num_local_embedding] : indices[tid + 1];
    int num_bucket_id = 0;
    for (int i = idx; i < next_idx; ++i) {
      if (sorted_dp_bucket_id_list[i] < batch_size * num_embedding) num_bucket_id += 1;
    }
    bucket_offset[tid + 1] = num_bucket_id;
  }

  if (tid == 0) {
    dst_idx[0] = 0;
    bucket_offset[0] = 0;
    unique_id_space_offset[num_local_embedding] = static_cast<uint32_t>(*num_indices);
  }
}

__global__ void memset_kernel(uint32_t* arr, int num, uint32_t val) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num; tid += blockDim.x * gridDim.x) {
    arr[tid] = val;
  }
}
}  // namespace

DPIndexCalculation::DPIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                       int num_local_embedding, int local_hotness_sum,
                                       int hotness_sum, int universal_batch_size, DataType key_type,
                                       DataType offset_type)
    : core_(core),
      num_gpus_(num_gpus),
      num_local_embedding_(num_local_embedding),
      universal_batch_size_(universal_batch_size),
      universal_batch_size_per_gpu_(universal_batch_size / num_gpus),
      local_hotness_sum_(local_hotness_sum),
      hotness_sum_(hotness_sum),
      key_type_(key_type),
      offset_type_(offset_type) {
  HugeCTR::CudaDeviceContext ctx(core->get_device_id());
  Device device{DeviceType::GPU, core->get_device_id()};

  core::BufferPtr buffer_ptr = GetBuffer(core_);

  // reserve and allocate tensors for index calculation on GPU
  num_dp_key_ = buffer_ptr->reserve({1}, DeviceType::CPU, TensorScalarType::Size_t);
  flag_ =
      buffer_ptr->reserve({universal_batch_size_ * hotness_sum_}, device, TensorScalarType::Char);

  size_t temp_storage_bytes_category = 0;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, ([&] {
                               cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes_category,
                                                          (key_t*)nullptr, (char*)nullptr,
                                                          (key_t*)nullptr, (size_t*)nullptr,
                                                          universal_batch_size_ * hotness_sum_);
                             }));
  d_temp_storage_category_ =
      buffer_ptr->reserve({temp_storage_bytes_category}, device, TensorScalarType::Void);

  size_t temp_storage_bytes_offset = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes_offset, (uint32_t*)nullptr,
                                (uint32_t*)nullptr,
                                universal_batch_size_per_gpu_ * num_local_embedding_ + 1);
  d_temp_storage_offset_ =
      buffer_ptr->reserve({temp_storage_bytes_offset}, device, TensorScalarType::Void);

  // allocate output memory
  dp_key_ =
      buffer_ptr->reserve({universal_batch_size_per_gpu_ * local_hotness_sum_}, device, key_type_);
  dp_offset_ = buffer_ptr->reserve({universal_batch_size_per_gpu_ * num_local_embedding_ + 1},
                                   device, TensorScalarType::UInt32);
  dp_dst_ = buffer_ptr->reserve({universal_batch_size_per_gpu_ * num_local_embedding_}, device,
                                TensorScalarType::UInt32);
  buffer_ptr->allocate();
}

void DPIndexCalculation::compute(const Tensor& key, const Tensor& bucket_range, size_t num_keys,
                                 const Tensor& d_local_embedding_list, int batch_size,
                                 Tensor* dp_key, Tensor* dp_offset, size_t* num_dp_key,
                                 Tensor* dp_dst) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  int batch_size_per_gpu = batch_size / num_gpus_;

  int gpu_id = core_->get_global_gpu_id();
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION(key.dtype().type(), key_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
      HCTR_LIB_THROW(cudaMemsetAsync(dp_key_.get<key_t>(), 0, dp_key_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(dp_offset_.get<uint32_t>(), 0, dp_offset_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(dp_dst_.get<uint32_t>(), 0, dp_dst_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(flag_.get<char>(), 0, flag_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(num_dp_key_.get<size_t>(), 0, num_dp_key_.nbytes(), stream));

      // mask_flag
      constexpr int blockDim = 1024;
      int gridDim = (num_local_embedding_ * batch_size_per_gpu - 1) / blockDim + 1;
      mask_flag_kernel<<<gridDim, blockDim, 0, stream>>>(
          num_local_embedding_, batch_size_per_gpu, d_local_embedding_list.get<int>(), batch_size,
          gpu_id, bucket_range.get<offset_t>(), flag_.get<char>(), dp_offset_.get<uint32_t>(),
          dp_dst_.get<uint32_t>());

      // select key
      size_t temp_storage_category_bytes = d_temp_storage_category_.nbytes();
      cub::DeviceSelect::Flagged(d_temp_storage_category_.get(), temp_storage_category_bytes,
                                 key.get<key_t>(), flag_.get<char>(), dp_key_.get<key_t>(),
                                 num_dp_key_.get<size_t>(), key.get_num_elements(), stream);
      HCTR_LIB_THROW(cudaPeekAtLastError());
      // inclusive sum for offset
      size_t temp_storage_offset_bytes = d_temp_storage_offset_.nbytes();
      cub::DeviceScan::InclusiveSum(d_temp_storage_offset_.get(), temp_storage_offset_bytes,
                                    dp_offset_.get<uint32_t>(), dp_offset_.get<uint32_t>(),
                                    dp_offset_.get_num_elements(), stream);
      HCTR_LIB_THROW(cudaPeekAtLastError());
      // sync with cpu to get sum flag
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  });

  *dp_key = dp_key_;
  *dp_offset = dp_offset_;
  *num_dp_key = num_dp_key_.get<size_t>()[0];
  *dp_dst = dp_dst_;
}

DPLocalReduceIndexCalculation::DPLocalReduceIndexCalculation(
    std::shared_ptr<CoreResourceManager> core, int num_embedding, int num_local_embedding,
    const std::vector<int>& h_local_hotness_list, int universal_batch_size, DataType key_type)
    : core_(core), num_embedding_(num_embedding), num_local_embedding_(num_local_embedding) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  Device device{DeviceType::GPU, core_->get_device_id()};

  int local_hotness_sum =
      std::accumulate(h_local_hotness_list.begin(), h_local_hotness_list.end(), 0);

  auto buffer_ptr = GetBuffer(core);

  segment_start_offsets_ =
      buffer_ptr->reserve(num_local_embedding + 1, device, TensorScalarType::Int32);
  segment_end_offsets_ = buffer_ptr->reserve(num_local_embedding, device, TensorScalarType::Int32);
  dp_keys_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device, key_type);
  dp_bucket_id_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                      TensorScalarType::UInt32);
  sorted_dp_keys_ =
      buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device, key_type);
  unique_dp_keys_indices_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                                TensorScalarType::UInt32);
  sorted_dp_bucket_id_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                             TensorScalarType::UInt32);

  unique_dp_keys_ =
      buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device, key_type);
  num_unique_key_ = buffer_ptr->reserve(1, DeviceType::CPU, TensorScalarType::Size_t);

  sorted_bucket_id_list_ = buffer_ptr->reserve({universal_batch_size, local_hotness_sum}, device,
                                               TensorScalarType::UInt32);
  num_sorted_bucket_id_ = buffer_ptr->reserve(1, DeviceType::CPU, TensorScalarType::Size_t);
  unique_dst_idx_ = buffer_ptr->reserve(1 + universal_batch_size * local_hotness_sum, device,
                                        TensorScalarType::UInt32);
  sorted_bucket_id_offset_ = buffer_ptr->reserve(1 + universal_batch_size * local_hotness_sum,
                                                 device, TensorScalarType::UInt32);
  unique_id_space_offset_ =
      buffer_ptr->reserve(1 + num_local_embedding_, device, TensorScalarType::UInt32);

  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
      cub::DeviceSegmentedRadixSort::SortPairs(
          nullptr, temp_bytes, (key_t*)nullptr, (key_t*)nullptr, (uint32_t*)nullptr,
          (uint32_t*)nullptr, universal_batch_size * local_hotness_sum, num_local_embedding_,
          (int*)nullptr, (int*)nullptr);
    });
    d_temp_segmented_sort_storage_ =
        buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
      cub::CountingInputIterator<uint32_t> counting(0);
      SelectUniqueDPKeyOp<key_t> select_unique_dp_key_op{nullptr, nullptr, nullptr,
                                                         num_local_embedding_};
      cub::DeviceSelect::If(nullptr, temp_bytes, counting, (uint32_t*)nullptr, (size_t*)nullptr,
                            universal_batch_size * local_hotness_sum, select_unique_dp_key_op);
    });
    d_temp_if_storage_ = buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
      SelectLocalBucketidOP select_unique_dp_key_op{num_local_embedding_ * universal_batch_size};
      cub::DeviceSelect::If(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                            (size_t*)nullptr, universal_batch_size * local_hotness_sum,
                            select_unique_dp_key_op);
    });
    d_temp_select_bucket_id_storage_ =
        buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  universal_batch_size * local_hotness_sum + 1);
    d_scan_storage_ = buffer_ptr->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  buffer_ptr->allocate();
  std::vector<int> cpu_segments_start_offset{0};
  for (int embedding_id = 0; embedding_id < num_local_embedding_; ++embedding_id) {
    cpu_segments_start_offset.push_back(h_local_hotness_list[embedding_id] * universal_batch_size);
  }
  std::partial_sum(cpu_segments_start_offset.begin(), cpu_segments_start_offset.end(),
                   cpu_segments_start_offset.begin());

  segment_start_offsets_.copy_from(cpu_segments_start_offset);
}

void DPLocalReduceIndexCalculation::compute(
    const Tensor& key, size_t num_key, const Tensor& bucket_range,
    const Tensor& d_local_embedding_list, const Tensor& id_space_list,
    const Tensor& d_local_ev_size_list, int batch_size, Tensor* unique_key, size_t* num_unique_key,
    Tensor* unique_dst_idx, Tensor* sorted_bucket_id_list, Tensor* sorted_bucket_id_offset,
    Tensor* unique_id_space_offset) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  DISPATCH_INTEGRAL_FUNCTION(key.dtype().type(), key_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();
      int gpu_id = core_->get_global_gpu_id();
      int num_gpus = core_->get_global_gpu_count();

      HCTR_LIB_THROW(cudaMemsetAsync(segment_end_offsets_.get<int>(), 0,
                                     segment_end_offsets_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(dp_keys_.get<key_t>(), 0, dp_keys_.nbytes(), stream));
      HCTR_LIB_THROW(
          cudaMemsetAsync(dp_bucket_id_.get<uint32_t>(), 0, dp_bucket_id_.nbytes(), stream));
      HCTR_LIB_THROW(
          cudaMemsetAsync(sorted_dp_keys_.get<key_t>(), 0, sorted_dp_keys_.nbytes(), stream));
      HCTR_LIB_THROW(
          cudaMemsetAsync(unique_dp_keys_.get<key_t>(), 0, unique_dp_keys_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(unique_dp_keys_indices_.get<uint32_t>(), 0,
                                     unique_dp_keys_indices_.nbytes(), stream));

      HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_list_.get<uint32_t>(), 0,
                                     sorted_bucket_id_list_.nbytes(), stream));
      HCTR_LIB_THROW(
          cudaMemsetAsync(unique_dst_idx_.get<uint32_t>(), 0, unique_dst_idx_.nbytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(sorted_bucket_id_offset_.get<uint32_t>(), 0,
                                     sorted_bucket_id_offset_.nbytes(), stream));

      {
        fused_select_dp_key_and_bucket_id_kernel<<<num_local_embedding_, 256, 0, stream>>>(
            key.get<key_t>(), bucket_range.get<offset_t>(), num_key, batch_size,
            num_local_embedding_, num_embedding_, gpu_id, num_gpus, id_space_list.get<int>(),
            d_local_embedding_list.get<int>(), dp_keys_.get<key_t>(), dp_bucket_id_.get<uint32_t>(),
            segment_start_offsets_.get<int>(), segment_end_offsets_.get<int>());
      }
      {
        memset_kernel<<<128, 1024, 0, stream>>>(sorted_dp_bucket_id_.get<uint32_t>(),
                                                sorted_dp_bucket_id_.get_num_elements(),
                                                batch_size * num_embedding_);
        size_t nbytes = d_temp_segmented_sort_storage_.nbytes();
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_segmented_sort_storage_.get(), nbytes, dp_keys_.get<key_t>(),
            sorted_dp_keys_.get<key_t>(), dp_bucket_id_.get<uint32_t>(),
            sorted_dp_bucket_id_.get<uint32_t>(), dp_keys_.get_num_elements(), num_local_embedding_,
            segment_start_offsets_.get<int>(), segment_end_offsets_.get<int>(), 0,
            sizeof(key_t) * 8, stream);
      }
      {
        cub::CountingInputIterator<uint32_t> counting(0);
        SelectUniqueDPKeyOp<key_t> select_unique_dp_key_op{
            sorted_dp_keys_.get<key_t>(), segment_start_offsets_.get<int>(),
            segment_end_offsets_.get<int>(), num_local_embedding_};
        size_t nbytes = d_temp_if_storage_.nbytes();
        cub::DeviceSelect::If(d_temp_if_storage_.get(), nbytes, counting,
                              unique_dp_keys_indices_.get<uint32_t>(),
                              num_unique_key_.get<size_t>(), sorted_dp_keys_.get_num_elements(),
                              select_unique_dp_key_op, stream);
      }
      {
        cub::CountingInputIterator<uint32_t> counting(0);
        SelectLocalBucketidOP select_unique_dp_key_op{num_embedding_ * batch_size};
        size_t nbytes = d_temp_if_storage_.nbytes();
        cub::DeviceSelect::If(
            d_temp_if_storage_.get(), nbytes, sorted_dp_bucket_id_.get<uint32_t>(),
            sorted_bucket_id_list_.get<uint32_t>(), num_sorted_bucket_id_.get<size_t>(),
            sorted_dp_bucket_id_.get_num_elements(), select_unique_dp_key_op, stream);

        fused_compact_unique_key_and_count_bucket_id_offset<<<(num_key - 1) / 256 + 1, 256, 0,
                                                              stream>>>(
            sorted_dp_keys_.get<key_t>(), unique_dp_keys_indices_.get<uint32_t>(),
            num_unique_key_.get<size_t>(), sorted_dp_bucket_id_.get<uint32_t>(),
            segment_start_offsets_.get<int>(), segment_end_offsets_.get<int>(),
            d_local_ev_size_list.get<int>(), batch_size, num_local_embedding_, num_embedding_,
            unique_dp_keys_.get<key_t>(), unique_dst_idx_.get<uint32_t>(),
            sorted_bucket_id_offset_.get<uint32_t>(), unique_id_space_offset_.get<uint32_t>());
      }
      {
        size_t nbytes = d_scan_storage_.nbytes();
        cub::DeviceScan::InclusiveSum(
            d_scan_storage_.get(), nbytes, unique_dst_idx_.get<uint32_t>(),
            unique_dst_idx_.get<uint32_t>(), unique_dst_idx_.get_num_elements(), stream);
        cub::DeviceScan::InclusiveSum(d_scan_storage_.get(), nbytes,
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get<uint32_t>(),
                                      sorted_bucket_id_offset_.get_num_elements(), stream);
      }
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  });
  *unique_key = unique_dp_keys_;
  *num_unique_key = *num_unique_key_.get<size_t>();
  *unique_dst_idx = unique_dst_idx_;
  *sorted_bucket_id_list = sorted_bucket_id_list_;
  *sorted_bucket_id_offset = sorted_bucket_id_offset_;
  *unique_id_space_offset = unique_id_space_offset_;
}
}  // namespace embedding