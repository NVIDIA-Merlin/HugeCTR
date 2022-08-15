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

#include "common/check.h"
#include "lookup/impl/select_kernel.h"

namespace sok {

template <typename KeyType>
__global__ void selectKernel(const KeyType *input_keys, size_t num_keys, KeyType *output_keys,
                             int32_t *output_indices, size_t chunks, size_t max_chunk_size,
                             int32_t *chunk_sizes, const size_t ITEMS_PER_GPU_PER_WARP,
                             const size_t KEY_WARPS_PER_BLOCK) {
  // set indices
  const size_t thread_cnt = blockDim.x * blockDim.y;
  const size_t stride_size = thread_cnt * gridDim.x;
  const size_t items_per_warp = chunks * ITEMS_PER_GPU_PER_WARP;
  const size_t items_per_block = KEY_WARPS_PER_BLOCK * items_per_warp;
  const size_t gpu_cnt_by_warps_cnt = chunks * KEY_WARPS_PER_BLOCK;
  const size_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
  // set ptrs in smem
  extern __shared__ char smem[];
  KeyType *key_smem = (KeyType *)smem;
  uint32_t *idx_smem = (uint32_t *)(key_smem + items_per_block);
  uint32_t *cnt_smem = idx_smem + items_per_block;
  if (thread_idx < gpu_cnt_by_warps_cnt) {
    cnt_smem[thread_idx] = 0;
  }
  // if (thread_idx + blockIdx.x * thread_cnt < chunks) {
  //   chunk_sizes[thread_idx] = 0;
  // }
  __syncthreads();
  // do offset
  KeyType *curr_warp_key_smem = key_smem + threadIdx.y * items_per_warp;
  uint32_t *curr_warp_idx_smem = idx_smem + threadIdx.y * items_per_warp;
  uint32_t *curr_warp_cnt_smem = cnt_smem + threadIdx.y * chunks;
  uint32_t padded_input_size = (num_keys + warpSize - 1) / warpSize * warpSize;
  // loop on input_keys
  for (size_t idx = thread_idx + blockIdx.x * thread_cnt; idx < padded_input_size;
       idx += stride_size) {
    KeyType key = 0;
    size_t chunk_id = 0;
    uint32_t curr_local_idx = 0;
    uint32_t offset = 0;
    uint32_t is_full = 0;
    if (idx < num_keys) {
      key = input_keys[idx];
      chunk_id = key % chunks;
      curr_local_idx = atomicAdd(curr_warp_cnt_smem + chunk_id, 1);
      offset = chunk_id * ITEMS_PER_GPU_PER_WARP + curr_local_idx;
      curr_warp_key_smem[offset] = key;
      curr_warp_idx_smem[offset] = idx;
    }
    is_full = (curr_local_idx == ITEMS_PER_GPU_PER_WARP - warpSize);
    uint32_t ballot_val = __ballot_sync(0xffffffff, is_full);
    // __syncwarp();
    int leading_zeros = __clz(ballot_val);
    while (leading_zeros < warpSize) {
      uint32_t full_gpu_idx = __shfl_sync(0xffffffff, chunk_id, warpSize - leading_zeros - 1);
      ballot_val &= (((uint32_t)0xffffffff) >> (leading_zeros + 1));
      leading_zeros = __clz(ballot_val);
      uint32_t curr_global_idx = 0;
      if (threadIdx.x == 0) {
        curr_global_idx = atomicAdd(chunk_sizes + full_gpu_idx, curr_warp_cnt_smem[full_gpu_idx]);
      }
      curr_global_idx = __shfl_sync(0xffffffff, curr_global_idx, 0);
      // __syncwarp();
      for (size_t output_idx = threadIdx.x; output_idx < curr_warp_cnt_smem[full_gpu_idx];
           output_idx += warpSize) {
        output_keys[full_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
            curr_warp_key_smem[full_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
        output_indices[full_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
            curr_warp_idx_smem[full_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
      }
      // __syncwarp();
    }
    __syncwarp();
    if (is_full) {
      curr_warp_cnt_smem[chunk_id] = 0;
    }
    __syncwarp();
  }
  // tail
  for (size_t has_gpu_idx = 0; has_gpu_idx < chunks; ++has_gpu_idx) {
    uint32_t curr_gpu_items = curr_warp_cnt_smem[has_gpu_idx];
    if (curr_gpu_items == 0) {
      continue;
    }
    uint32_t curr_global_idx = 0;
    if (threadIdx.x == 0) {
      curr_global_idx = atomicAdd(chunk_sizes + has_gpu_idx, curr_warp_cnt_smem[has_gpu_idx]);
    }
    curr_global_idx = __shfl_sync(0xffffffff, curr_global_idx, 0);
    for (size_t output_idx = threadIdx.x; output_idx < curr_warp_cnt_smem[has_gpu_idx];
         output_idx += warpSize) {
      output_keys[has_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
          curr_warp_key_smem[has_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
      output_indices[has_gpu_idx * max_chunk_size + curr_global_idx + output_idx] =
          curr_warp_idx_smem[has_gpu_idx * ITEMS_PER_GPU_PER_WARP + output_idx];
    }
    __syncwarp();
  }
}

template <typename KeyType>
void SelectLauncher<KeyType>::initialize(size_t num_splits) {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
  CUDACHECK(cudaDeviceGetAttribute(&warp_size_, cudaDevAttrWarpSize, device));
  CUDACHECK(cudaDeviceGetAttribute(&max_shared_memory_per_sm_,
                                   cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  max_shared_memory_per_sm_ -= (4 * 1024);
  CUDACHECK(cudaFuncSetAttribute(selectKernel<KeyType>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 max_shared_memory_per_sm_));
  key_warps_per_block_ = 8;
  items_per_gpu_per_warp_ =
      max_shared_memory_per_sm_ - (sizeof(uint32_t) * key_warps_per_block_ * num_splits);
  items_per_gpu_per_warp_ /=
      (num_splits * key_warps_per_block_ * (sizeof(KeyType) + sizeof(uint32_t)));
  items_per_gpu_per_warp_ = (items_per_gpu_per_warp_ / 16) * 16;
  host_splits_.resize(num_splits);
}

template <typename KeyType>
void SelectLauncher<KeyType>::operator()(const void *indices, size_t num_keys, void *output,
                                         void *output_buffer, void *order, void *order_buffer,
                                         void *splits, size_t num_splits, cudaStream_t stream) {
  const KeyType *t_indices = reinterpret_cast<const KeyType *>(indices);
  KeyType *t_output = reinterpret_cast<KeyType *>(output);
  KeyType *t_output_buffer = reinterpret_cast<KeyType *>(output_buffer);
  int32_t *t_order = reinterpret_cast<int32_t *>(order);
  int32_t *t_order_buffer = reinterpret_cast<int32_t *>(order_buffer);
  int32_t *t_splits = reinterpret_cast<int32_t *>(splits);

  // Launch kernel
  CUDACHECK(cudaMemsetAsync(t_splits, 0, sizeof(int32_t) * num_splits, stream));
  size_t grid_dim = sm_count_;
  dim3 block_dim(warp_size_, key_warps_per_block_);
  selectKernel<KeyType><<<grid_dim, block_dim, max_shared_memory_per_sm_, stream>>>(
      t_indices, num_keys, t_output_buffer, t_order_buffer, num_splits, num_keys, t_splits,
      items_per_gpu_per_warp_, key_warps_per_block_);
  CUDACHECK(cudaGetLastError());

  // Copy data from output_buffer & order_buffer to output & order.
  CUDACHECK(cudaMemcpyAsync(host_splits_.data(), t_splits, sizeof(int32_t) * num_splits,
                            cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  size_t offset = 0;
  for (int i = 0; i < num_splits; ++i) {
    CUDACHECK(cudaMemcpyAsync(t_output + offset, t_output_buffer + i * num_keys,
                              sizeof(KeyType) * host_splits_[i], cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(t_order + offset, t_order_buffer + i * num_keys,
                              sizeof(int32_t) * host_splits_[i], cudaMemcpyDeviceToDevice, stream));
    offset += host_splits_[i];
  }
}

template class SelectLauncher<int64_t>;
template class SelectLauncher<int32_t>;

}  // namespace sok
