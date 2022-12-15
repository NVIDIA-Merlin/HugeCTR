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

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <cub/cub.cuh>
#include <memory>
#include <vector>

#include "HugeCTR/include/common.hpp"

namespace HugeCTR {
namespace DeviceSelect {
namespace detail {
template <typename T, typename IndexType, typename SelectOp, int BlockSize = 1024>
__global__ void pre_select_if(const T *d_input, unsigned short *d_offset, IndexType *d_block_sum,
                              size_t len, SelectOp op, T *d_num_selected_out = nullptr) {
  unsigned short this_thread_sum = 0;
  unsigned short this_block_sum = 0;

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  size_t gtid = static_cast<size_t>(blockIdx.x) * BlockSize + static_cast<size_t>(threadIdx.x);
  // a trick to
  if (!gtid) {
    *(d_block_sum - 1) = 0;
  }
  if (gtid < len) {
    IndexType in = d_input ? static_cast<IndexType>(d_input[gtid]) : static_cast<IndexType>(gtid);
    this_thread_sum = static_cast<unsigned short>(op(in));
  }
  __syncthreads();
  typedef cub::BlockScan<unsigned short, BlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan(temp_storage).ExclusiveSum(this_thread_sum, this_thread_sum, this_block_sum);
  __syncthreads();
  if (tid == 0) {
    d_block_sum[bid] = static_cast<IndexType>(this_block_sum);
  }
  if (gtid < len) {
    d_offset[gtid] = this_thread_sum;
  }
}
template <typename T, typename IndexType, typename SelectOp, int BlockSize = 1024>
__global__ void post_select_if(const T *d_input, const unsigned short *d_offset,
                               const IndexType *d_block_offset, size_t len, SelectOp Op, T *output,
                               T *d_num_selected_out) {
  int64_t global_index = 0;
  __shared__ IndexType src_data[BlockSize];

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  size_t gtid = static_cast<size_t>(blockIdx.x) * BlockSize + static_cast<size_t>(threadIdx.x);
  if (gtid < len) {
    // d_offset + d_block_offset to get the global index
    global_index = static_cast<int64_t>(d_block_offset[bid] + static_cast<int64_t>(d_offset[gtid]));
    // vectorized load
    IndexType in = d_input ? static_cast<IndexType>(d_input[gtid]) : static_cast<IndexType>(gtid);
    src_data[tid] = in;
  }
  __syncthreads();
  // warp divergence
  if (gtid < len && Op(src_data[tid])) {
    output[global_index] = src_data[tid];
  }
  if (!gtid) {
    *d_num_selected_out = d_block_offset[gridDim.x];
  }
};
}  // namespace detail

template <typename T, typename IndexType, typename InputIteratorT, typename SelectOp>
void If(void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT input, T *output,
        T *d_num_selected_out, IndexType num_items, SelectOp Op, cudaStream_t stream = 0) {
  constexpr unsigned int blocksize = 1024;
  unsigned int gridDim = (num_items - 1) / (blocksize) + 1;
  using cubCountIt = cub::CountingInputIterator<T>;
  const T *input_ptr{nullptr};
  if constexpr (!std::is_same<InputIteratorT, cubCountIt>::value) {
    input_ptr = reinterpret_cast<const T *>(input);
  }

  if (!d_temp_storage) {
    temp_storage_bytes = 0;
    temp_storage_bytes += sizeof(IndexType) * (gridDim + 1);
    temp_storage_bytes += sizeof(unsigned short) * (num_items);
    size_t cub_bytes = 0;
    HCTR_LIB_THROW(cub::DeviceScan::InclusiveSum((void *)(nullptr), cub_bytes,
                                                 (IndexType *)(nullptr), (IndexType *)(nullptr),
                                                 gridDim, stream));
    temp_storage_bytes += cub_bytes;
    return;
  }
  size_t temp_start = reinterpret_cast<size_t>(d_temp_storage);
  IndexType *d_block_sum = reinterpret_cast<IndexType *>(temp_start);
  temp_start += sizeof(IndexType) * (gridDim + 1);
  unsigned short *d_offset = reinterpret_cast<unsigned short *>(temp_start);
  temp_start += sizeof(unsigned short) * (num_items);
  size_t cub_bytes = temp_storage_bytes + reinterpret_cast<size_t>(d_temp_storage) -
                     reinterpret_cast<size_t>(temp_start);
  detail::pre_select_if<<<gridDim, blocksize, 0, stream>>>(
      input_ptr, d_offset, d_block_sum + 1, (size_t)num_items, Op, d_num_selected_out);
  HCTR_LIB_THROW(cub::DeviceScan::InclusiveSum(reinterpret_cast<void *>(temp_start), cub_bytes,
                                               d_block_sum + 1, d_block_sum + 1, gridDim, stream));
  detail::post_select_if<<<gridDim, blocksize, 0, stream>>>(
      input_ptr, d_offset, d_block_sum, (size_t)num_items, Op, output, d_num_selected_out);
}
}  // namespace DeviceSelect
}  // namespace HugeCTR