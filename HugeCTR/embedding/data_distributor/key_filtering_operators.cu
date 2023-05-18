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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

#include <utils.cuh>
#include <utils.hpp>

#include "key_filtering_operators.hpp"

namespace cg = cooperative_groups;

namespace HugeCTR {

using namespace mp;
using namespace dp;

namespace kernels {

template <typename key_t, typename offset_t>
__global__ void label_and_count_keys(
    const key_t** __restrict keys, const offset_t** __restrict bucket_range,
    const int* __restrict lookup_bucket_threads, const int* __restrict lookup_ids,
    const int* __restrict lookup_num_shards, const int* __restrict lookup_gpus,
    const int* __restrict hotness_bucket_range, const uint32_t* __restrict gpu_lookup_range,
    key_t* flat_keys, uint32_t* labels, offset_t* keys_per_gpu, offset_t* keys_per_bucket,
    int batch_size_per_gpu, int num_gpus, int this_gpu_id, int num_lookup) {
  extern __shared__ uint32_t dynamic_smem[];
  uint32_t* smem_keys_per_gpu = dynamic_smem;
  uint32_t* smem_lookup_gpus = dynamic_smem + num_gpus;

  int shard_id = blockIdx.y;
  int lookup = lookup_ids[shard_id];

  const key_t* lookup_keys = keys[lookup];
  const offset_t* lookup_bucket_range = bucket_range[lookup];
  int num_shards = lookup_num_shards[lookup];
  int bucket_threads = lookup_bucket_threads[lookup];
  int lookup_bucket_start = hotness_bucket_range[lookup] * batch_size_per_gpu;

  labels += lookup_bucket_start;  // skip to where we care about
  flat_keys += lookup_bucket_start;

  for (int i = threadIdx.x; i < num_shards; i += blockDim.x) {
    smem_lookup_gpus[i] = lookup_gpus[lookup * num_gpus + i];
  }

  for (int i = threadIdx.x; i < num_gpus; i += blockDim.x) {
    smem_keys_per_gpu[i] = 0;
  }

  __syncthreads();

  cg::thread_group bucket_group = cg::tiled_partition(cg::this_thread_block(), bucket_threads);
  int cta_num_groups = blockDim.x / bucket_group.size();
  int cta_group_id = threadIdx.x / bucket_group.size();

  // max_hotness * num_threads < INT16/INT
  // num_threads * num_global_gpus // 128 * 128 * sizeof(int) = 64K
  // num_threads * num_lookup_gpus =  128 * 16 * sizeof(int) =

  for (int bucket = blockIdx.x * cta_num_groups + cta_group_id; bucket < batch_size_per_gpu;
       bucket += gridDim.x * cta_num_groups) {
    // Rely on L1 cache
    offset_t range_start = lookup_bucket_range[bucket];
    offset_t range_end = lookup_bucket_range[bucket + 1];
    int num_keys = static_cast<int>(range_end - range_start);

    for (int k = bucket_group.thread_rank(); k < num_keys; k += bucket_group.size()) {
      // Which GPU does my embedding reside on?
      key_t key = lookup_keys[range_start + k];
      int gpu_id = smem_lookup_gpus[key % num_shards];
      labels[range_start + k] = gpu_id;
      flat_keys[range_start + k] = key;

      atomic_add(&smem_keys_per_gpu[gpu_id], 1);

      // Rely on L1, TODO: shared?
      auto lookup_start_range = gpu_lookup_range[gpu_id * num_lookup + lookup];

      atomic_add(&keys_per_bucket[lookup_start_range + bucket], static_cast<offset_t>(1));
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_gpus; i += blockDim.x) {
    atomic_add(&keys_per_gpu[i], smem_keys_per_gpu[i]);
  }
}

/**
 * Converts array of gpu-major buckets to feature-major buckets
 */
template <typename offset_t>
__global__ void transpose_buckets(const offset_t* __restrict buckets, offset_t* buckets_transposed,
                                  int global_gpu_count, int num_shards, int batch_size_per_gpu) {
  int shard_id = blockIdx.y;
  assert(shard_id < num_shards);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size_per_gpu;
       i += blockDim.x * gridDim.x) {
    for (int gpu_id = 0; gpu_id < global_gpu_count; ++gpu_id) {
      int r_idx = (gpu_id * batch_size_per_gpu * num_shards) + (shard_id * batch_size_per_gpu + i);
      int w_idx =
          (shard_id * global_gpu_count * batch_size_per_gpu) + (gpu_id * batch_size_per_gpu + i);
      assert(r_idx < num_shards * global_gpu_count * batch_size_per_gpu);
      assert(w_idx < num_shards * global_gpu_count * batch_size_per_gpu);
      buckets_transposed[w_idx] = buckets[r_idx];
    }
  }
}

template <typename key_t, typename offset_t>
__global__ void swizzle_keys(const offset_t* __restrict src_bucket_range,
                             const offset_t* __restrict dst_bucket_range,
                             const key_t* __restrict keys, key_t* swizzled_keys,
                             const int* __restrict shard_bucket_threads, int batch_size_per_gpu,
                             int num_shards, int global_gpu_count) {
  int shard_id = blockIdx.y;
  assert(shard_id < num_shards);

  int bucket_threads = shard_bucket_threads[shard_id];

  cg::thread_group bucket_group = cg::tiled_partition(cg::this_thread_block(), bucket_threads);

  int num_groups = blockDim.x / bucket_group.size();
  int group_id = threadIdx.x / bucket_group.size();

  for (int i = blockIdx.x * num_groups + group_id; i < batch_size_per_gpu * global_gpu_count;
       i += gridDim.x * num_groups) {
    int gpu_id = i / batch_size_per_gpu;
    int lookup_bucket_id = i - (gpu_id * batch_size_per_gpu);

    int src_bucket_idx = (gpu_id * batch_size_per_gpu * num_shards) +
                         (shard_id * batch_size_per_gpu + lookup_bucket_id);
    int dst_bucket_idx = (shard_id * global_gpu_count * batch_size_per_gpu) + i;

    // Rely on L1 cache
    offset_t dst_range_start = dst_bucket_range[dst_bucket_idx];
    offset_t src_range_start = src_bucket_range[src_bucket_idx];
    offset_t src_range_end = src_bucket_range[src_bucket_idx + 1];

    int num_keys = static_cast<int>(src_range_end - src_range_start);

    for (int k = bucket_group.thread_rank(); k < num_keys; k += bucket_group.size()) {
      swizzled_keys[dst_range_start + k] = keys[src_range_start + k];
    }
  }
}

// warp reduce via shfl -> cta reduce via smem atomic -> global reduce via L2/dram atomic
template <typename offset_t>
__global__ void count_keys_per_gpu(const offset_t* __restrict k_per_b_gpu_major,
                                   offset_t* keys_per_gpu, int num_buckets_per_dev,
                                   int num_buckets_per_dev_padded) {
  int gpu_id = blockIdx.y;

  k_per_b_gpu_major += gpu_id * num_buckets_per_dev;

  __shared__ offset_t smem_keys_per_gpu;
  if (threadIdx.x == 0) smem_keys_per_gpu = static_cast<offset_t>(0);
  __syncthreads();

  auto tile = cg::tiled_partition<32>(cg::this_thread_block());

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_buckets_per_dev_padded;
       i += blockDim.x * gridDim.x) {
    int bucket_num_keys = k_per_b_gpu_major[min(i, num_buckets_per_dev - 1)];
    bucket_num_keys = i >= num_buckets_per_dev ? 0 : bucket_num_keys;
    int sum = cg::reduce(tile, bucket_num_keys, cg::plus<int>());
    //    printf("gpu_id: %d, idx: %d, bucket_num_keys: %d, sum: %d\n", gpu_id, i, bucket_num_keys,
    //    sum);
    if (tile.thread_rank() == 0) {
      atomic_add(&smem_keys_per_gpu, sum);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomic_add(&keys_per_gpu[gpu_id], smem_keys_per_gpu);
  }
}

template <typename offset_t>
__global__ void compute_bucket_ranges_with_padding(offset_t** bucket_ranges,
                                                   offset_t* keys_per_bucket,
                                                   const int* __restrict max_hotnesses,
                                                   int local_batch_size, int batch_size_per_gpu) {
  const int lookup_id = blockIdx.y;
  const int hotness = max_hotnesses[lookup_id];
  auto bucket_range = bucket_ranges[lookup_id];

  // e.g:
  // max_hotnesses:      [3, 5, 1, 2]
  // batch_size_per_gpu: 5, local_batch_size: 3
  // bucket_range[0]: [0,3,6,9,9,9]
  const offset_t end_range = local_batch_size * hotness;

  // extend batch_size by 1 to account for end bucket
  //  CUDA_1D_KERNEL_LOOP(bucket_idx, batch_size_per_gpu + 1) {
  //    const bool is_valid_bucket = bucket_idx < local_batch_size + 1;
  //    bucket_range[bucket_idx] =
  //        is_valid_bucket ? static_cast<offset_t>(bucket_idx * hotness) : end_range;
  //  }

  CUDA_1D_KERNEL_LOOP(bucket_idx, batch_size_per_gpu) {
    const bool is_valid_bucket = bucket_idx < local_batch_size + 1;
    bucket_range[bucket_idx] =
        is_valid_bucket ? static_cast<offset_t>(bucket_idx * hotness) : end_range;
    keys_per_bucket[lookup_id * batch_size_per_gpu + bucket_idx] =
        bucket_idx < local_batch_size ? hotness : 0;
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int bucket_idx = batch_size_per_gpu;
    const bool is_valid_bucket = bucket_idx < local_batch_size + 1;
    bucket_range[bucket_idx] =
        is_valid_bucket ? static_cast<offset_t>(bucket_idx * hotness) : end_range;
  }
}

template <int TILE_SIZE, typename offset_t>
__global__ void compute_shard_ranges(uint32_t* shard_ranges,
                                     const offset_t** __restrict bucket_ranges,
                                     const int* shard_ids, int num_shards, int num_shards_padded,
                                     int batch_size_per_gpu) {
  assert(gridDim.x == 1 && gridDim.y == 1);  // only need 1 CTA for this

  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block);

  __shared__ uint32_t scan_start;

  auto tid = tile.thread_rank();

  if (tid == 0) {
    shard_ranges[tid] = 0;
    scan_start = 0;
  }

  for (int i = tid; i < num_shards_padded; i += tile.size()) {
    tile.sync();

    uint32_t num_keys = 0;
    if (i < num_shards) {
      int shard_id = shard_ids[i];
      num_keys = static_cast<uint32_t>(bucket_ranges[shard_id][batch_size_per_gpu]);
    }

    uint32_t scan_out = scan_start + cg::inclusive_scan(tile, num_keys);

    if (i < num_shards) {
      shard_ranges[i + 1] = scan_out;
    }

    if (tid == tile.size() - 1) {
      scan_start = scan_out;
    }
  }
}

template <typename key_t, typename offset_t>
__global__ void concat_keys_and_bucket_range(key_t* concat_keys, offset_t* concat_bucket_ranges,
                                             const key_t** __restrict dp_keys,
                                             const offset_t** __restrict dp_bucket_ranges,
                                             const int* shard_ids,
                                             const uint32_t* __restrict shard_ranges,
                                             const int* __restrict shard_bucket_threads,
                                             int batch_size_per_gpu) {
  int i = blockIdx.y;
  int shard_id = shard_ids[i];
  const offset_t* shard_bucket_range = dp_bucket_ranges[shard_id];
  const key_t* shard_keys = dp_keys[shard_id];
  offset_t offset = static_cast<offset_t>(shard_ranges[i]);
  int bucket_threads = shard_bucket_threads[i];

  int end_bucket = i == (gridDim.y - 1);

  cg::thread_group bucket_group = cg::tiled_partition(cg::this_thread_block(), bucket_threads);

  int num_groups = blockDim.x / bucket_group.size();
  int group_id = threadIdx.x / bucket_group.size();

  for (int bucket_idx = blockIdx.x * num_groups + group_id; bucket_idx < batch_size_per_gpu;
       bucket_idx += gridDim.x * num_groups) {
    // Rely on L1 cache
    offset_t start_range = shard_bucket_range[bucket_idx];
    offset_t end_range = shard_bucket_range[bucket_idx + 1];
    int num_keys = static_cast<int>(end_range - start_range);

    for (int k = bucket_group.thread_rank(); k < num_keys; k += bucket_group.size()) {
      concat_keys[offset + start_range + k] = shard_keys[start_range + k];
    }

    if (bucket_group.thread_rank() == 0) {
      concat_bucket_ranges[i * batch_size_per_gpu + bucket_idx] = offset + start_range;
    }
  }

  if (end_bucket && blockIdx.x == 0 && threadIdx.x == 0) {
    int bucket_idx = batch_size_per_gpu;
    offset_t start_range = shard_bucket_range[bucket_idx];
    concat_bucket_ranges[i * batch_size_per_gpu + bucket_idx] = offset + start_range;
  }
}

}  // namespace kernels

template <typename T>
T round_up(T x, T y) {
  return ((x + y - 1) / y) * y;
}

static int highest_pow2(int n) {
  int p = (int)log2(n);
  int r = (int)pow(2, p);
  assert((r & (r - 1)) == 0 && "result not power of 2");
  return r;
}

LabelAndCountKeysOperator::LabelAndCountKeysOperator(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id)
    : batch_size_(ebc_param.universal_batch_size),
      batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()),
      global_gpu_id_(core->get_global_gpu_id()),
      global_gpu_count_(core->get_global_gpu_count()),
      core_(core) {
  CudaDeviceContext ctx(core->get_device_id());

  std::vector<int> h_lookup_gpu_ids(ebc_param.num_lookup * global_gpu_count_, 0);
  std::vector<int> h_lookup_num_gpus(ebc_param.num_lookup, 0);
  std::vector<int> h_lookup_bucket_threads(ebc_param.num_lookup, 0);
  std::vector<int> h_hotness_bucket_range(ebc_param.num_lookup + 1, 0);
  std::vector<int> h_lookup_ids;
  // e.g: num_tables = 4, GPU0 tables [0, 2], GPU1 tables [1, 2, 3]
  //        gpu_lookup_range = [0,1,1,2,2|2,2,3,4,5] * batch_size
  h_per_gpu_lookup_range = std::vector<uint32_t>(global_gpu_count_ * ebc_param.num_lookup + 1, 0);

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    bool lookup_in_group = false;

    for (int gpu_id = 0; gpu_id < global_gpu_count_; ++gpu_id) {
      if (ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) {
        h_lookup_gpu_ids[lookup_id * global_gpu_count_ + h_lookup_num_gpus[lookup_id]] = gpu_id;
        h_lookup_num_gpus[lookup_id]++;
        h_per_gpu_lookup_range[gpu_id * ebc_param.num_lookup + lookup_id + 1] = batch_size_per_gpu_;
        lookup_in_group = true;
      }
    }

    if (lookup_in_group) h_lookup_ids.push_back(lookup_id);

    // cooperative groups need partition to be power of 2
    int hotness = ebc_param.lookup_params[lookup_id].max_hotness;
    h_lookup_bucket_threads[lookup_id] = min(highest_pow2(hotness), 32);

    h_hotness_bucket_range[lookup_id + 1] = hotness;
  }

  std::inclusive_scan(h_per_gpu_lookup_range.begin() + 1, h_per_gpu_lookup_range.end(),
                      h_per_gpu_lookup_range.begin() + 1);
  std::inclusive_scan(h_hotness_bucket_range.begin() + 1, h_hotness_bucket_range.end(),
                      h_hotness_bucket_range.begin() + 1);

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->lookup_gpu_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_gpu_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  this->lookup_num_gpus =
      core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_num_gpus.size())})
                         .data_type(core23::ScalarType::Int32));
  this->lookup_bucket_threads =
      core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_bucket_threads.size())})
                         .data_type(core23::ScalarType::Int32));
  this->hotness_bucket_range =
      core23::Tensor(params.shape({static_cast<int64_t>(h_hotness_bucket_range.size())})
                         .data_type(core23::ScalarType::Int32));
  this->gpu_lookup_range =
      core23::Tensor(params.shape({static_cast<int64_t>(h_per_gpu_lookup_range.size())})
                         .data_type(core23::ScalarType::UInt32));
  this->lookup_ids = core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_ids.size())})
                                        .data_type(core23::ScalarType::Int32));

  core23::copy_sync(lookup_gpu_ids, h_lookup_gpu_ids);
  core23::copy_sync(lookup_num_gpus, h_lookup_num_gpus);
  core23::copy_sync(lookup_bucket_threads, h_lookup_bucket_threads);
  core23::copy_sync(hotness_bucket_range, h_hotness_bucket_range);
  core23::copy_sync(gpu_lookup_range, h_per_gpu_lookup_range);
  core23::copy_sync(lookup_ids, h_lookup_ids);
}

void LabelAndCountKeysOperator::operator()(const DataDistributionInput& input,
                                           LabelAndCountKeysOperator::Result& output,
                                           cudaStream_t stream) {
  HCTR_LIB_THROW(cudaMemsetAsync(output.keys_per_bucket.data(), 0,
                                 output.keys_per_bucket.num_bytes(), stream));
  HCTR_LIB_THROW(
      cudaMemsetAsync(output.keys_per_gpu.data(), 0, output.keys_per_gpu.num_bytes(), stream));

  // Set labels to max so when we process incomplete batch the invalid keys are set to the max value
  // and will be sorted correctly
  HCTR_LIB_THROW(
      cudaMemsetAsync(output.local_labels.data(), 0xFF, output.local_labels.num_bytes(), stream));

  dim3 block(128);
  dim3 grid((batch_size_per_gpu_ + block.x - 1) / block.x, lookup_ids.num_elements());

  int smem_lookup_gpus_bytes = global_gpu_count_ * sizeof(int);
  int smem_keys_per_gpu = global_gpu_count_ * sizeof(uint32_t);
  int smem_bytes = smem_keys_per_gpu + smem_lookup_gpus_bytes;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(input.key_type.type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(input.offset_type.type(), BucketRangeType, [&] {
      kernels::label_and_count_keys<<<grid, block, smem_bytes, stream>>>(
          input.get_dp_keys_pointer_ptr<KeyType>(),
          input.get_dp_bucket_range_pointer_ptr<BucketRangeType>(),
          (const int*)lookup_bucket_threads.data<int>(), (const int*)lookup_ids.data<int>(),
          (const int*)lookup_num_gpus.data<int>(), (const int*)lookup_gpu_ids.data<int>(),
          (const int*)hotness_bucket_range.data<int>(),
          (const uint32_t*)gpu_lookup_range.data<uint32_t>(), output.flat_keys.data<KeyType>(),
          output.local_labels.data<uint32_t>(), output.keys_per_gpu.data<BucketRangeType>(),
          output.keys_per_bucket.data<BucketRangeType>(), batch_size_per_gpu_, global_gpu_count_,
          global_gpu_id_, input.num_lookup_);
    });
  });

  HCTR_LIB_THROW(cudaGetLastError());
}

LabelAndCountKeysOperator::Result::Result(
    std::shared_ptr<core::CoreResourceManager> resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id) {
  CudaDeviceContext ctx(resource_manager->get_device_id());

  const auto& lookup_params = ebc_param.lookup_params;
  const auto& group_params = ebc_param.grouped_table_params[grouped_id];
  const int batch_size_per_gpu =
      ebc_param.universal_batch_size / resource_manager->get_global_gpu_count();

  size_t num_features = 0;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    num_features += ebc_param.lookup_params[lookup_id].max_hotness;
  }

  size_t num_buckets = 0;
  for (size_t gpu_id = 0; gpu_id < resource_manager->get_global_gpu_count(); ++gpu_id) {
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      if (ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) {
        num_buckets += batch_size_per_gpu;
      }
    }
  }

  core23::Device device(core23::DeviceType::GPU, resource_manager->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->local_labels =
      core23::Tensor(params.shape({static_cast<int64_t>(num_features * batch_size_per_gpu)})
                         .data_type(core23::ScalarType::UInt32));
  this->keys_per_bucket = core23::Tensor(
      params.shape({static_cast<int64_t>(num_buckets)}).data_type(ebc_param.offset_type));
  this->keys_per_gpu =
      core23::Tensor(params.shape({static_cast<int64_t>(resource_manager->get_global_gpu_count())})
                         .data_type(ebc_param.offset_type));
  this->flat_keys =
      core23::Tensor(params.shape({static_cast<int64_t>(num_features * batch_size_per_gpu)})
                         .data_type(ebc_param.key_type.type()));
}

CountKeysOperator::CountKeysOperator(std::shared_ptr<core::CoreResourceManager> core,
                                     const embedding::EmbeddingCollectionParam& ebc_param,
                                     size_t grouped_id)
    : batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()),
      global_gpu_count_(core->get_global_gpu_count()),
      num_shards_(0) {
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (ebc_param.has_table_shard(core->get_global_gpu_id(), grouped_id, lookup_id)) {
      num_shards_++;
    }
  }
}

void CountKeysOperator::operator()(core23::Tensor keys_per_bucket_gpu_major,
                                   core23::Tensor result_keys_per_gpu, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((batch_size_per_gpu_ * num_shards_ + block.x - 1) / block.x, global_gpu_count_);

  int num_buckets_per_dev = num_shards_ * batch_size_per_gpu_;
  int num_buckets_per_dev_padded = round_up(num_buckets_per_dev, 32);  // rounded up to warp size

  HCTR_LIB_THROW(
      cudaMemsetAsync(result_keys_per_gpu.data(), 0, result_keys_per_gpu.num_bytes(), stream));
  if (num_shards_ == 0) return;

  HCTR_CHECK(keys_per_bucket_gpu_major.data_type() == result_keys_per_gpu.data_type());
  DISPATCH_INTEGRAL_FUNCTION_CORE23(
      keys_per_bucket_gpu_major.data_type().type(), BucketRangeType, [&] {
        kernels::count_keys_per_gpu<<<grid, block, sizeof(BucketRangeType), stream>>>(
            keys_per_bucket_gpu_major.data<BucketRangeType>(),
            result_keys_per_gpu.data<BucketRangeType>(), num_buckets_per_dev,
            num_buckets_per_dev_padded);
      });
  HCTR_LIB_THROW(cudaGetLastError());
}

TransposeBucketsOperator::TransposeBucketsOperator(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id)
    : num_shards_(0),
      batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()),
      global_gpu_count_(core->get_global_gpu_count()) {
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (ebc_param.has_table_shard(core->get_global_gpu_id(), grouped_id, lookup_id)) {
      num_shards_++;
    }
  }
}

void TransposeBucketsOperator::operator()(core23::Tensor buckets_gpu_major,
                                          core23::Tensor buckets_feat_major, cudaStream_t stream) {
  if (num_shards_ == 0) return;

  dim3 block(128);
  dim3 grid((batch_size_per_gpu_ + block.x - 1) / block.x, num_shards_);

  DISPATCH_INTEGRAL_FUNCTION_CORE23(buckets_gpu_major.data_type().type(), BucketRangeType, [&] {
    kernels::transpose_buckets<<<grid, block, 0, stream>>>(
        buckets_gpu_major.data<BucketRangeType>(), buckets_feat_major.data<BucketRangeType>(),
        global_gpu_count_, num_shards_, batch_size_per_gpu_);
  });
  HCTR_LIB_THROW(cudaGetLastError());
}

SwizzleKeysOperator::SwizzleKeysOperator(std::shared_ptr<core::CoreResourceManager> core,
                                         const embedding::EmbeddingCollectionParam& ebc_param,
                                         size_t grouped_id)
    : num_shards_(0),
      batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()),
      global_gpu_count_(core->get_global_gpu_count()) {
  std::vector<int> h_shard_bucket_threads;

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (ebc_param.has_table_shard(core->get_global_gpu_id(), grouped_id, lookup_id)) {
      // cooperative groups need partition to be power of 2
      int hotness = ebc_param.lookup_params[lookup_id].max_hotness;
      h_shard_bucket_threads.push_back(1 /*highest_pow2(hotness)*/);
      num_shards_++;
    }
  }

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  this->shard_bucket_threads_ = core23::Tensor(
      params.shape({static_cast<int64_t>(num_shards_)}).data_type(core23::ScalarType::Int32));

  core23::copy_sync(shard_bucket_threads_, h_shard_bucket_threads);
}

void SwizzleKeysOperator::operator()(core23::Tensor src_bucket_range,
                                     core23::Tensor dst_bucket_range, core23::Tensor keys,
                                     core23::Tensor result_keys, cudaStream_t stream) {
  if (num_shards_ == 0) return;

  dim3 block(128);
  dim3 grid((batch_size_per_gpu_ * global_gpu_count_ + block.x - 1) / block.x, num_shards_);

  DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(src_bucket_range.data_type().type(), BucketRangeType, [&] {
      kernels::swizzle_keys<<<grid, block, 0, stream>>>(
          src_bucket_range.data<BucketRangeType>(), dst_bucket_range.data<BucketRangeType>(),
          keys.data<KeyType>(), result_keys.data<KeyType>(), shard_bucket_threads_.data<int>(),
          batch_size_per_gpu_, num_shards_, global_gpu_count_);
    });
  });
  HCTR_LIB_THROW(cudaGetLastError());
}

ComputeDPBucketRangeOperator::ComputeDPBucketRangeOperator(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param)
    : global_gpu_id_(core->get_global_gpu_id()),
      batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()) {
  CudaDeviceContext ctx(core->get_device_id());

  std::vector<int> h_max_hotnesses(ebc_param.num_lookup, 0);
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    h_max_hotnesses[lookup_id] = ebc_param.lookup_params[lookup_id].max_hotness;
  }

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->max_hotnesses_ = core23::Tensor(params.shape({static_cast<int64_t>(ebc_param.num_lookup)})
                                            .data_type(core23::ScalarType::Int32));
  core23::copy_sync(max_hotnesses_, h_max_hotnesses);

  d_ptrs_ = core23::Tensor(params.shape({static_cast<int64_t>(ebc_param.num_lookup)})
                               .data_type(core23::ScalarType::Pointer));
  h_ptrs_ = core23::Tensor(core23::TensorParams()
                               .device(core23::DeviceType::CPU)
                               .shape({static_cast<int64_t>(ebc_param.num_lookup)})
                               .data_type(core23::ScalarType::Pointer));
}

void ComputeDPBucketRangeOperator::operator()(std::vector<core23::Tensor> dp_bucket_ranges,
                                              core23::Tensor keys_per_bucket,
                                              int current_batch_size, cudaStream_t stream) {
  int remaining = current_batch_size - global_gpu_id_ * batch_size_per_gpu_;
  int num_valid_samples = std::max(std::min(remaining, batch_size_per_gpu_), 0);
  int num_lookup = dp_bucket_ranges.size();

  init_tensor_list(h_ptrs_, dp_bucket_ranges);
  core23::copy_async(d_ptrs_, h_ptrs_, stream);

  dim3 block(128);
  dim3 grid((batch_size_per_gpu_ + block.x - 1) / block.x, num_lookup);

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_bucket_ranges[0].data_type().type(), BucketRangeType, [&] {
    kernels::compute_bucket_ranges_with_padding<<<grid, block, 0, stream>>>(
        d_ptrs_.data<BucketRangeType*>(), keys_per_bucket.data<BucketRangeType>(),
        max_hotnesses_.data<int>(), num_valid_samples, batch_size_per_gpu_);
  });

  HCTR_LIB_THROW(cudaGetLastError());
}

ConcatKeysAndBucketRangeOperator::ConcatKeysAndBucketRangeOperator(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id)
    : batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()) {
  CudaDeviceContext ctx(core->get_device_id());

  std::vector<int> h_shard_bucket_threads;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (ebc_param.has_table_shard(core->get_global_gpu_id(), grouped_id, lookup_id)) {
      h_shard_ids_.push_back(lookup_id);

      // cooperative groups need partition to be power of 2
      int hotness = ebc_param.lookup_params[lookup_id].max_hotness;
      h_shard_bucket_threads.push_back(min(highest_pow2(hotness), 32));
    }
  }

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  d_shard_ids_ = core23::Tensor(params.shape({static_cast<int64_t>(h_shard_ids_.size())})
                                    .data_type(core23::ScalarType::Int32));
  shard_bucket_threads_ = core23::Tensor(params.shape({static_cast<int64_t>(h_shard_ids_.size())})
                                             .data_type(core23::ScalarType::Int32));
  shard_ranges_ = core23::Tensor(params.shape({static_cast<int64_t>(h_shard_ids_.size()) + 1})
                                     .data_type(core23::ScalarType::UInt32));

  core23::copy_sync(d_shard_ids_, h_shard_ids_);
  core23::copy_sync(shard_bucket_threads_, h_shard_bucket_threads);
}

void ConcatKeysAndBucketRangeOperator::operator()(const DataDistributionInput& input,
                                                  core23::Tensor& result_keys,
                                                  core23::Tensor& result_bucket_range,
                                                  cudaStream_t stream) {
  int num_shards = h_shard_ids_.size();

  auto key_scalar_type = input.key_type.type();
  auto range_scalar_type = input.offset_type.type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(range_scalar_type, BucketRangeType, [&] {
    constexpr int block_dim = 32;
    int num_shards_padded = ((num_shards + block_dim - 1) / block_dim) * block_dim;

    kernels::compute_shard_ranges<block_dim><<<1, block_dim, 0, stream>>>(
        shard_ranges_.data<uint32_t>(), input.get_dp_bucket_range_pointer_ptr<BucketRangeType>(),
        d_shard_ids_.data<int>(), num_shards, num_shards_padded, batch_size_per_gpu_);
  });
  HCTR_LIB_THROW(cudaGetLastError());

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_scalar_type, KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(range_scalar_type, BucketRangeType, [&] {
      dim3 block(128);
      dim3 grid((batch_size_per_gpu_ + block.x - 1) / block.x, num_shards);

      kernels::concat_keys_and_bucket_range<<<grid, block, 0, stream>>>(
          result_keys.data<KeyType>(), result_bucket_range.data<BucketRangeType>(),
          input.get_dp_keys_pointer_ptr<KeyType>(),
          input.get_dp_bucket_range_pointer_ptr<BucketRangeType>(), d_shard_ids_.data<int>(),
          shard_ranges_.data<uint32_t>(), shard_bucket_threads_.data<int>(), batch_size_per_gpu_);
    });
  });
  HCTR_LIB_THROW(cudaGetLastError());
}

}  // namespace HugeCTR
