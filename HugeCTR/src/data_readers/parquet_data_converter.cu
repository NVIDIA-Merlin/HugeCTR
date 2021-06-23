/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <inttypes.h>

#include <cstring>
#include <cub/cub.cuh>
#include <deque>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/resource_manager.hpp"

namespace HugeCTR {

__device__ __forceinline__ unsigned int __mylaneid() {
  unsigned int laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}

__device__ __forceinline__ unsigned int abs(unsigned int x) { return x; }

template <typename T>
__global__ void dense_data_converter_kernel__(int64_t *dense_data_column_ptrs,
                                              const int label_dense_dim, int batch_size,
                                              int num_dense_buffers, int64_t *dense_data_out_ptrs) {
  // extern __shared__ char smem_[];
  // 8 warps/block
  int tile_w = 32;  // 32x32 tile
  int smem_pitch = 33;
  __shared__ T smem_staging_ptr[8 * 32 * 33];

  int start_idx = threadIdx.x + (blockDim.x * blockIdx.x);

  // outer loop on label_dense_dim
  for (int i = 0; i < label_dense_dim; i += warpSize) {
    // stage 32x32 tile - stage 32 columns of data
    // warpSize drives the row dim to 32
    for (int j = 0; j < tile_w; j++) {
      // warp does one column at a time
      int col = i + j;
      if (col < label_dense_dim) {
        int64_t addr = dense_data_column_ptrs[col];
        T *data = reinterpret_cast<T *>(addr);
        if (start_idx < batch_size) {
          smem_staging_ptr[threadIdx.x * smem_pitch + j] = data[start_idx];
        }
      }
    }
    __syncthreads();

    // write out

    int out_row_idx = __shfl_sync(0xffffffff, start_idx, 0);
    // each warp writes out 32 rows and whatever active columns in j(32)
    if ((__mylaneid() + i) < label_dense_dim) {
      // activate threads

      // blockStrided warp over 32 rows write out
      int warp_id = threadIdx.x / warpSize;
      int smem_row = warp_id * warpSize;

      // warpsize doing tile_h
      for (int j = 0; j < warpSize; j++) {
        if ((j + out_row_idx) < batch_size) {
          int curr_out_row = j + out_row_idx;
          int buffer_id = curr_out_row / (batch_size / num_dense_buffers);
          int local_id = curr_out_row % (batch_size / num_dense_buffers);

          int64_t addr = dense_data_out_ptrs[buffer_id];
          T *data = reinterpret_cast<T *>(addr);

          data[(local_id * label_dense_dim) + __mylaneid() + i] =
              smem_staging_ptr[(smem_row + j) * smem_pitch + __mylaneid()];
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void cat_local_slot_converter_kernel__(int64_t *cat_data_column_ptrs, int num_params,
                                                  int param_id, int num_slots, int batch_size,
                                                  int num_devices, int32_t *dev_slot_per_device,
                                                  T *dev_slot_offset_ptr,
                                                  int64_t *dev_csr_value_ptr,
                                                  int64_t *dev_csr_row_offset_ptr,
                                                  uint32_t *dev_csr_row_offset_counter) {
  // 8 warps/block
  int tile_w = 32;  // 32x32 tile
  int smem_pitch = 33;
  extern __shared__ char smem_[];
  T *smem_staging_ptr = reinterpret_cast<T *>(smem_);
  int num_warps = blockDim.x / warpSize;
  uint32_t *block_atomic_accum =
      reinterpret_cast<uint32_t *>(smem_ + (num_warps * warpSize * smem_pitch * sizeof(T)));

  // zero out smem_accum
  for (int i = threadIdx.x; i < num_params * num_devices; i += blockDim.x)
    block_atomic_accum[i] = 0;
  __syncthreads();

  int start_idx = threadIdx.x + (blockDim.x * blockIdx.x);

  for (int i = 0; i < num_slots; i += warpSize) {
    // stage 32x32 tile - stage 32 columns of data
    // warpSize drives the row dim to 32
    for (int j = 0; j < tile_w; j++) {
      // warp does one column at a time
      int col = i + j;
      if (col < num_slots) {
        int64_t addr = cat_data_column_ptrs[col];
        T *data = reinterpret_cast<T *>(addr);
        if (start_idx < batch_size) {
          smem_staging_ptr[threadIdx.x * smem_pitch + j] =
              data[start_idx] + dev_slot_offset_ptr[col];
        }
      }
    }
    __syncthreads();

    // write out

    int out_row_idx = __shfl_sync(0xffffffff, start_idx, 0);
    // each warp writes out 32 rows and whatever active columns in j(32)
    if ((__mylaneid() + i) < num_slots) {
      // activate threads

      // blockStrided warp over 32 rows write out
      int warp_id = threadIdx.x / warpSize;
      int smem_row = warp_id * warpSize;

      // warpsize doing tile_h
      for (int j = 0; j < warpSize; j++) {
        if ((j + out_row_idx) < batch_size) {
          int curr_out_row = j + out_row_idx;  // batch n

          // select buffer:, use dev_slot_per_device for row in that and value buffer
          int slot_id = (__mylaneid() + i);
          int dev_id = slot_id % num_devices;
          int buffer_id = dev_id * num_params + param_id;
          int local_id = (slot_id - dev_id) / num_devices;

          int64_t addr = dev_csr_value_ptr[buffer_id];  // dev_csr_value_ptr
          T *data = reinterpret_cast<T *>(addr);
          uint32_t idx_to_buffers = curr_out_row * dev_slot_per_device[dev_id] + local_id;
          data[idx_to_buffers] = smem_staging_ptr[(smem_row + j) * smem_pitch + __mylaneid()];

          int64_t addr_row_buf = dev_csr_row_offset_ptr[buffer_id];
          T *row_offset = reinterpret_cast<T *>(addr_row_buf);
          row_offset[idx_to_buffers] = 1;

          atomicAdd(&(block_atomic_accum[buffer_id]), 1);
        }
      }
    }
    __syncthreads();
  }
  // update global atomic buffer
  for (int i = threadIdx.x; i < num_params * num_devices; i += blockDim.x) {
    atomicAdd(&(dev_csr_row_offset_counter[i]), block_atomic_accum[i]);
  }
}

template <typename T>
__global__ void cat_distributed_slot_csr_roffset_kernel__(int64_t *cat_data_column_ptrs,
                                                          int num_params, int param_id,
                                                          int num_slots, int batch_size,
                                                          int num_devices, T *dev_slot_offset_ptr,
                                                          int64_t *dev_csr_row_offset_ptr,
                                                          uint32_t *dev_csr_row_offset_counter) {
  // 8 warps/block
  int tile_w = 32;  // 32x32 tile
  int smem_pitch = 33;
  extern __shared__ char smem_[];
  T *smem_staging_ptr = reinterpret_cast<T *>(smem_);
  int num_warps = blockDim.x / warpSize;
  uint32_t *block_atomic_accum =
      reinterpret_cast<uint32_t *>(smem_ + (num_warps * warpSize * smem_pitch * sizeof(T)));

  // zero out smem_accum
  for (int i = threadIdx.x; i < num_params * num_devices; i += blockDim.x)
    block_atomic_accum[i] = 0;
  __syncthreads();

  int start_idx = threadIdx.x + (blockDim.x * blockIdx.x);

  for (int i = 0; i < num_slots; i += warpSize) {
    // stage 32x32 tile - stage 32 columns of data
    // warpSize drives the row dim to 32
    for (int j = 0; j < tile_w; j++) {
      // warp does one column at a time
      int col = i + j;
      if (col < num_slots) {
        int64_t addr = cat_data_column_ptrs[col];
        T *data = reinterpret_cast<T *>(addr);
        if (start_idx < batch_size) {
          smem_staging_ptr[threadIdx.x * smem_pitch + j] =
              data[start_idx] + dev_slot_offset_ptr[col];
        }
      }
    }
    __syncthreads();

    // write out

    int out_row_idx = __shfl_sync(0xffffffff, start_idx, 0);
    // each warp writes out 32 rows and whatever active columns in j(32)
    if ((__mylaneid() + i) < num_slots) {
      // activate threads

      // blockStrided warp over 32 rows write out
      int warp_id = threadIdx.x / warpSize;
      int smem_row = warp_id * warpSize;

      // warpsize doing tile_h
      for (int j = 0; j < warpSize; j++) {
        if ((j + out_row_idx) < batch_size) {
          int curr_out_row = j + out_row_idx;  // batch n

          T val = smem_staging_ptr[(smem_row + j) * smem_pitch + __mylaneid()];
          int dev_id = abs(val % num_devices);
          int buffer_id = dev_id * num_params + param_id;
          int slot_id = (__mylaneid() + i);

          // adjust this with prev param count as well
          uint32_t idx_to_buffers = curr_out_row * num_slots + slot_id;

          // dont need this if using staging for initial idx markers
          // idx_to_buffers += param_offset_buf[buffer_id];  // TBD

          int64_t addr_row_buf = dev_csr_row_offset_ptr[buffer_id];
          T *row_offset = reinterpret_cast<T *>(addr_row_buf);
          row_offset[idx_to_buffers] = 1;
          // row_offset[idx_to_buffers] = 1 + param_offset_buf[buffer_id];

          // for ex scan to get write values in distributed row_offset csr buffers
          if ((slot_id == (num_slots - 1)) && (curr_out_row == (batch_size - 1)))
            row_offset[idx_to_buffers + 1] = 1;

          atomicAdd(&(block_atomic_accum[buffer_id]), 1);
        }
      }
    }
    __syncthreads();
  }

  // update global atomic buffer
  for (int i = threadIdx.x; i < num_params * num_devices; i += blockDim.x) {
    atomicAdd(&(dev_csr_row_offset_counter[i]), block_atomic_accum[i]);
  }
}

template <typename T>
__global__ void cat_distributed_slot_csr_val_kernel__(int64_t *cat_data_column_ptrs, int num_params,
                                                      int param_id, int num_slots, int batch_size,
                                                      int num_devices, T *dev_slot_offset_ptr,
                                                      int64_t *dev_csr_row_val_ptr,
                                                      int64_t *dev_csr_row_offset_ptr) {
  // 8 warps/block
  int tile_w = 32;  // 32x32 tile
  int smem_pitch = 33;
  extern __shared__ char smem_[];
  T *smem_staging_ptr = reinterpret_cast<T *>(smem_);

  int start_idx = threadIdx.x + (blockDim.x * blockIdx.x);

  for (int i = 0; i < num_slots; i += warpSize) {
    // stage 32x32 tile - stage 32 columns of data
    // warpSize drives the row dim to 32
    for (int j = 0; j < tile_w; j++) {
      // warp does one column at a time
      int col = i + j;
      if (col < num_slots) {
        int64_t addr = cat_data_column_ptrs[col];
        T *data = reinterpret_cast<T *>(addr);
        if (start_idx < batch_size) {
          smem_staging_ptr[threadIdx.x * smem_pitch + j] =
              data[start_idx] + dev_slot_offset_ptr[col];
        }
      }
    }
    __syncthreads();

    // write out

    int out_row_idx = __shfl_sync(0xffffffff, start_idx, 0);
    // each warp writes out 32 rows and whatever active columns in j(32)
    if ((__mylaneid() + i) < num_slots) {
      // activate threads

      // blockStrided warp over 32 rows write out
      int warp_id = threadIdx.x / warpSize;
      int smem_row = warp_id * warpSize;

      // warpsize doing tile_h
      for (int j = 0; j < warpSize; j++) {
        if ((j + out_row_idx) < batch_size) {
          int curr_out_row = j + out_row_idx;  // batch n

          T val = smem_staging_ptr[(smem_row + j) * smem_pitch + __mylaneid()];
          int dev_id = abs(val % num_devices);
          int buffer_id = dev_id * num_params + param_id;
          int slot_id = (__mylaneid() + i);

          uint32_t idx_to_buffers = curr_out_row * num_slots + slot_id;
          int64_t addr_row_buf = dev_csr_row_offset_ptr[buffer_id];
          T *row_offset = reinterpret_cast<T *>(addr_row_buf);
          T idx = row_offset[idx_to_buffers];

          int64_t addr_val_buf = dev_csr_row_val_ptr[buffer_id];
          T *val_buf = reinterpret_cast<T *>(addr_val_buf);
          val_buf[idx] = val;
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void check_and_set_csr_row_kernel_(int64_t *dev_csr_row_offset_ptr,
                                              uint32_t *dev_csr_row_offset_counter,
                                              int max_elements_csr_row, int buffer_id) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  int64_t addr = dev_csr_row_offset_ptr[buffer_id];
  T *data = reinterpret_cast<T *>(addr);

  if (idx < max_elements_csr_row) {
    T val = data[idx];
    if (val == 1) {
      data[idx] = (T)idx;
    } else if (idx == dev_csr_row_offset_counter[buffer_id]) {
      data[idx] = (T)idx;
    }
  }
}

/**
 * Interleave dense (continuous) data parquet columns and write to linear buffer
 * @param dense_column_data_ptr vector of device pointers to Parquet columns
 * @param label_dense_dim number of dense values
 * @param batch_size batch size to load
 * @param num_dense_buffers number of dense buffers
 * @param dense_data_buffers vector of device buffers to write output
 * @param dev_ptr_staging pointer to pinned memory for copying pointer address from h2d
 * @param rmm_resources Queue to hold reference to RMM allocations
 * @param mr Device memory resource for RMM allocations
 * @param task_stream Stream to allocate memory and launch kerenels
 */
template <typename T>
void convert_parquet_dense_columns(std::vector<T *> &dense_column_data_ptr,
                                   const int label_dense_dim, int batch_size, int num_dense_buffers,
                                   std::vector<rmm::device_buffer> &dense_data_buffers,
                                   int64_t *dev_ptr_staging,
                                   std::deque<rmm::device_buffer> &rmm_resources,
                                   rmm::mr::device_memory_resource *mr, cudaStream_t task_stream) {
  // tiled load and transpose
  size_t size_of_col_ptrs = dense_column_data_ptr.size() * sizeof(T *);
  std::memcpy(dev_ptr_staging, dense_column_data_ptr.data(), size_of_col_ptrs);

  rmm_resources.emplace_back(size_of_col_ptrs, task_stream, mr);
  rmm::device_buffer &dev_in_column_ptr = rmm_resources.back();
  CK_CUDA_THROW_(cudaMemcpyAsync(dev_in_column_ptr.data(), dev_ptr_staging, size_of_col_ptrs,
                                 cudaMemcpyHostToDevice, task_stream));

  int64_t *pinned_dev_out_buffer =
      reinterpret_cast<int64_t *>((size_t)(dev_ptr_staging) + size_of_col_ptrs);
  for (unsigned int i = 0; i < dense_data_buffers.size(); i++) {
    pinned_dev_out_buffer[i] = (int64_t)dense_data_buffers[i].data();
  }

  size_t size_of_out_ptrs = dense_data_buffers.size() * sizeof(int64_t);

  rmm_resources.emplace_back(size_of_out_ptrs, task_stream, mr);
  rmm::device_buffer &dev_out_data_ptr = rmm_resources.back();

  CK_CUDA_THROW_(cudaMemcpyAsync(dev_out_data_ptr.data(), pinned_dev_out_buffer, size_of_out_ptrs,
                                 cudaMemcpyHostToDevice, task_stream));

  // assuming 48KB smem/SM
  // 32x32 tile per warp -> 4096 bytes/warp
  // 12 warps -> 384 threads/block
  // size_t smem_size = 48 * 1024 * 1024;
  dim3 block(256, 1, 1);
  dim3 grid((batch_size - 1) / block.x + 1, 1, 1);

  dense_data_converter_kernel__<T><<<grid, block, 0, task_stream>>>(
      (int64_t *)dev_in_column_ptr.data(), label_dense_dim, batch_size, num_dense_buffers,
      (int64_t *)dev_out_data_ptr.data());

  CK_CUDA_THROW_(cudaGetLastError());

  return;
}

/**
 * Interleave categoricals (slot) data parquet columns and write to csr buffers
 * @param cat_column_data_ptr vector of device pointers to Parquet columns
 * @param num_params number of Embedding params
 * @param param_id param idx for current param
 * @param num_slots number of slots in current param
 * @param batch_size batch size to load
 * @param num_csr_buffers number of csr buffers in csr heap
 * @param num_devices number of gpu devices
 * @param distributed_slot flag to set distributed slot processing
 * @param pid pid of node
 * @param resource_manager ResourceManager handle for session
 * @param csr_value_buffers vector of device buffers to write csr values
 * @param csr_row_offset_buffers vector of device buffers to write csr row offset values
 * @param dev_ptr_staging pointer to pinned memory for copying pointer address from h2d
 * @param dev_embed_param_offset_buf memory to atomically accumulate values written to csr val buf
 * @param dev_slot_offset_ptr device buffer with value for slot value offsets to make unique index
 * @param rmm_resources Queue to hold reference to RMM allocations
 * @param mr Device memory resource for RMM allocations
 * @param task_stream Stream to allocate memory and launch kerenels
 */
// for nnz =1 csr size_of_value and size_of_row_offset inc will be same
template <typename T>
size_t convert_parquet_cat_columns(std::vector<T *> &cat_column_data_ptr, int num_params,
                                   int param_id, int num_slots, int batch_size, int num_csr_buffers,
                                   int num_devices, bool distributed_slot, int pid,
                                   const std::shared_ptr<ResourceManager> resource_manager,
                                   std::vector<rmm::device_buffer> &csr_value_buffers,
                                   std::vector<rmm::device_buffer> &csr_row_offset_buffers,
                                   int64_t *dev_ptr_staging, uint32_t *dev_embed_param_offset_buf,
                                   T *dev_slot_offset_ptr,
                                   std::deque<rmm::device_buffer> &rmm_resources,
                                   rmm::mr::device_memory_resource *mr, cudaStream_t task_stream) {
  size_t pinned_staging_elements_used = 0;
  // tiled load and transpose
  size_t size_of_col_ptrs = cat_column_data_ptr.size() * sizeof(int64_t *);
  std::memcpy(dev_ptr_staging, cat_column_data_ptr.data(), size_of_col_ptrs);
  pinned_staging_elements_used += cat_column_data_ptr.size();

  rmm_resources.emplace_back(size_of_col_ptrs, task_stream, mr);
  rmm::device_buffer &dev_in_column_ptr = rmm_resources.back();
  CK_CUDA_THROW_(cudaMemcpyAsync(dev_in_column_ptr.data(), dev_ptr_staging, size_of_col_ptrs,
                                 cudaMemcpyHostToDevice, task_stream));

  size_t size_of_csr_pointers = num_csr_buffers * sizeof(int64_t);

  int64_t *pinned_csr_val_out_buffer =
      reinterpret_cast<int64_t *>((size_t)(dev_ptr_staging) + size_of_col_ptrs);
  for (int i = 0; i < num_csr_buffers; i++) {
    pinned_csr_val_out_buffer[i] = (int64_t)csr_value_buffers[i].data();
  }
  pinned_staging_elements_used += num_csr_buffers;

  rmm_resources.emplace_back(size_of_csr_pointers, task_stream, mr);
  rmm::device_buffer &dev_csr_value_ptr = rmm_resources.back();
  CK_CUDA_THROW_(cudaMemcpyAsync(dev_csr_value_ptr.data(), pinned_csr_val_out_buffer,
                                 size_of_csr_pointers, cudaMemcpyHostToDevice, task_stream));

  int64_t *pinned_csr_row_offset_buffer = reinterpret_cast<int64_t *>(
      (size_t)(dev_ptr_staging) + size_of_col_ptrs + size_of_csr_pointers);
  for (int i = 0; i < num_csr_buffers; i++) {
    pinned_csr_row_offset_buffer[i] = (int64_t)csr_row_offset_buffers[i].data();
  }
  pinned_staging_elements_used += num_csr_buffers;

  rmm_resources.emplace_back(size_of_csr_pointers, task_stream, mr);
  rmm::device_buffer &dev_csr_row_offset_ptr = rmm_resources.back();
  CK_CUDA_THROW_(cudaMemcpyAsync(dev_csr_row_offset_ptr.data(), pinned_csr_row_offset_buffer,
                                 size_of_csr_pointers, cudaMemcpyHostToDevice, task_stream));

  if (distributed_slot) {
    std::vector<rmm::device_buffer> csr_row_offset_staging;
    size_t csr_roff_buf_size = (size_t)((num_slots * batch_size + 1) * sizeof(T));
    for (int i = 0; i < num_csr_buffers; i++) {
      csr_row_offset_staging.emplace_back(csr_roff_buf_size, task_stream, mr);

      CK_CUDA_THROW_(
          cudaMemsetAsync(csr_row_offset_staging.back().data(), 0, csr_roff_buf_size, task_stream));
    }

    int64_t *pinned_csr_row_offset_staging =
        reinterpret_cast<int64_t *>((size_t)pinned_csr_row_offset_buffer + size_of_csr_pointers);

    for (int i = 0; i < num_csr_buffers; i++) {
      pinned_csr_row_offset_staging[i] = (int64_t)csr_row_offset_staging[i].data();
    }
    pinned_staging_elements_used += num_csr_buffers;

    rmm_resources.emplace_back(size_of_csr_pointers, task_stream, mr);
    rmm::device_buffer &dev_csr_row_offset_staging_ptr = rmm_resources.back();
    CK_CUDA_THROW_(cudaMemcpyAsync(dev_csr_row_offset_staging_ptr.data(),
                                   pinned_csr_row_offset_staging, size_of_csr_pointers,
                                   cudaMemcpyHostToDevice, task_stream));

    int block_size = (sizeof(T) == 8) ? 128 : 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((batch_size - 1) / block.x + 1, 1, 1);
    size_t smem_size = (block_size / 32) * sizeof(T) * 32 * 33;
    size_t smem_atomic_buffer = num_devices * num_params * sizeof(uint32_t);
    smem_size += smem_atomic_buffer;
    size_t max_smem_size = 48 * 1024;

    if (smem_size > max_smem_size)
      CK_THROW_(Error_t::OutOfMemory, "Parquet Converter: Not enough shared memory availble");

    // 2 -pass, setup row_offset, prefix_sum, write val to buf idx provided by prefix sum
    cat_distributed_slot_csr_roffset_kernel__<T><<<grid, block, smem_size, task_stream>>>(
        (int64_t *)dev_in_column_ptr.data(), num_params, param_id, num_slots, batch_size,
        num_devices, dev_slot_offset_ptr, (int64_t *)dev_csr_row_offset_staging_ptr.data(),
        dev_embed_param_offset_buf);

    CK_CUDA_THROW_(cudaGetLastError());
    for (int i = 0; i < num_csr_buffers; i++) {
      rmm_resources.emplace_back(std::move(csr_row_offset_staging.back()));
      csr_row_offset_staging.pop_back();
    }
    // prefix sum
    // dont really need to do prefix sum on int64 - check for future
    void *tmp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int prefix_sum_items = num_slots * batch_size + 1;
    CK_CUDA_THROW_(cub::DeviceScan::ExclusiveSum(
        tmp_storage, temp_storage_bytes, reinterpret_cast<T *>(pinned_csr_row_offset_staging[0]),
        reinterpret_cast<T *>(pinned_csr_row_offset_buffer[0]), prefix_sum_items, task_stream));

    rmm_resources.emplace_back(temp_storage_bytes, task_stream, mr);
    rmm::device_buffer &cub_tmp_storage = rmm_resources.back();

    /********************
    how to make prefix sum write at correct location??
    - you already incremented the staging with running atomic counter in kernel
    will exscan sum even work when buffers start rolling in ?
    may need inscan for param_id > 0 */

    // dont need all that per param csr buffers are different
    // exscan on only current param's csr buffers
    for (int i = 0; i < num_devices; i++) {
      if (pid == resource_manager->get_process_id_from_gpu_global_id(i)) {
        int buffer_id = i * num_params + param_id;
        CK_CUDA_THROW_(cub::DeviceScan::ExclusiveSum(
            cub_tmp_storage.data(), temp_storage_bytes,
            reinterpret_cast<T *>(pinned_csr_row_offset_staging[buffer_id]),
            reinterpret_cast<T *>(pinned_csr_row_offset_buffer[buffer_id]), prefix_sum_items,
            task_stream));
      } else {
        // pinned_csr_row_offset_buffer[x] are init'd to zero - no need to set again
      }
    }

    // kernel to set csr value based on idx generated in prefix scan - everything is single-hot
    cat_distributed_slot_csr_val_kernel__<T><<<grid, block, smem_size, task_stream>>>(
        (int64_t *)dev_in_column_ptr.data(), num_params, param_id, num_slots, batch_size,
        num_devices, dev_slot_offset_ptr, (int64_t *)dev_csr_value_ptr.data(),
        (int64_t *)dev_csr_row_offset_ptr.data());
    CK_CUDA_THROW_(cudaGetLastError());
  } else {
    int32_t *pinned_slot_per_device =
        reinterpret_cast<int32_t *>((size_t)pinned_csr_row_offset_buffer + size_of_csr_pointers);
    // localized embedding , generate slot to idx count mappings
    for (int i = 0; i < num_devices; i++) pinned_slot_per_device[i] = 0;

    for (int i = 0; i < num_slots; i++) {
      pinned_slot_per_device[i % num_devices]++;
    }
    pinned_staging_elements_used += num_devices;

    rmm_resources.emplace_back(num_devices * sizeof(int32_t), task_stream, mr);
    rmm::device_buffer &dev_slot_per_device_ptr = rmm_resources.back();

    CK_CUDA_THROW_(cudaMemcpyAsync(dev_slot_per_device_ptr.data(), pinned_slot_per_device,
                                   num_devices * sizeof(int32_t), cudaMemcpyHostToDevice,
                                   task_stream));

    int block_size = (sizeof(T) == 8) ? 128 : 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((batch_size - 1) / block.x + 1, 1, 1);
    size_t smem_size = (block_size / 32) * sizeof(T) * 32 * 33;
    size_t smem_atomic_buffer = num_devices * num_params * sizeof(uint32_t);
    smem_size += smem_atomic_buffer;
    size_t max_smem_size = 48 * 1024;

    if (smem_size > max_smem_size)
      CK_THROW_(Error_t::OutOfMemory, "Parquet Converter: Not enough shared memory availble");

    cat_local_slot_converter_kernel__<T><<<grid, block, smem_size, task_stream>>>(
        (int64_t *)dev_in_column_ptr.data(), num_params, param_id, num_slots, batch_size,
        num_devices, (int32_t *)dev_slot_per_device_ptr.data(), dev_slot_offset_ptr,
        (int64_t *)dev_csr_value_ptr.data(), (int64_t *)dev_csr_row_offset_ptr.data(),
        dev_embed_param_offset_buf);

    CK_CUDA_THROW_(cudaGetLastError());
    // csr_row_offset col val = idx
    // everything is single-hot , single-param for now
    // future - take-in atomic_offset_counter from last fn call to start at right offset of
    // csr_row_offset_buf same offset goes to converter_kernel as well for both value, row_offset
    // buffer

    int max_elements_csr_row = num_slots * batch_size + 1;
    dim3 block_2(1024, 1, 1);
    dim3 grid_2((max_elements_csr_row - 1) / block_2.x + 1, 1, 1);

    for (int device = 0; device < num_devices; device++) {
      if (pid == resource_manager->get_process_id_from_gpu_global_id(device)) {
        int buf_id = device * num_params + param_id;
        check_and_set_csr_row_kernel_<T><<<grid_2, block_2, 0, task_stream>>>(
            (int64_t *)dev_csr_row_offset_ptr.data(), dev_embed_param_offset_buf,
            max_elements_csr_row, buf_id);
      }
    }
  }

  CK_CUDA_THROW_(cudaGetLastError());

  return pinned_staging_elements_used;
}

// init function instances here
template void convert_parquet_dense_columns<float>(
    std::vector<float *> &dense_column_data_ptr, const int label_dense_dim, int batch_size,
    int num_dense_buffers, std::vector<rmm::device_buffer> &dense_data_buffers,
    int64_t *dev_ptr_staging, std::deque<rmm::device_buffer> &rmm_resources,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<long long int>(
    std::vector<long long int *> &cat_column_data_ptr, int num_params, int param_id, int num_slots,
    int batch_size, int num_csr_buffers, int num_devices, bool distributed_slot, int pid,
    const std::shared_ptr<ResourceManager> resource_manager,
    std::vector<rmm::device_buffer> &csr_value_buffers,
    std::vector<rmm::device_buffer> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    uint32_t *dev_embed_param_offset_buf, long long *dev_slot_offset_ptr,
    std::deque<rmm::device_buffer> &rmm_resources, rmm::mr::device_memory_resource *mr,
    cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<unsigned int>(
    std::vector<unsigned int *> &cat_column_data_ptr, int num_params, int param_id, int num_slots,
    int batch_size, int num_csr_buffers, int num_devices, bool distributed_slot, int pid,
    const std::shared_ptr<ResourceManager> resource_manager,
    std::vector<rmm::device_buffer> &csr_value_buffers,
    std::vector<rmm::device_buffer> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    uint32_t *dev_embed_param_offset_buf, unsigned int *dev_slot_offset_ptr,
    std::deque<rmm::device_buffer> &rmm_resources, rmm::mr::device_memory_resource *mr,
    cudaStream_t task_stream);

}  // namespace HugeCTR
