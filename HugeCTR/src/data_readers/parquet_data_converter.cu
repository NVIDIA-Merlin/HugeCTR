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

#include <common.hpp>
#include <cstring>
#include <cub/cub.cuh>
#include <deque>
#include <resource_manager.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <vector>

namespace HugeCTR {

__device__ __forceinline__ unsigned int __mylaneid() {
  unsigned int laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}
//  type of row_offset : int32_t
template <typename T>
__device__ __forceinline__ T *upper_bound(T *start, T *end, T target) {
  T left = 0;
  T right = end - start;

  while (left < right) {
    T mid = (left + right) >> 1;
    T val = *(start + mid);
    if (val > target) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  return start + left;
}

__device__ __forceinline__ unsigned int abs(unsigned int x) { return x; }

// reserved in case all dense are scalar
template <typename T>
__global__ void dense_data_converter_kernel_scalar_(int64_t *dense_data_column_ptrs,
                                                    const int label_dense_dim, int batch_size,
                                                    T *dense_data_out_ptrs) {
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
          int local_id = curr_out_row % (batch_size);
          T *data = dense_data_out_ptrs;

          data[(local_id * label_dense_dim) + __mylaneid() + i] =
              smem_staging_ptr[(smem_row + j) * smem_pitch + __mylaneid()];
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void dense_data_converter_kernel__(int64_t *dense_data_column_ptrs, const int num_dense,
                                              const int64_t *dense_dim_array,
                                              const int label_dense_dim, int batch_size,
                                              T *dense_data_out_ptrs) {
  int start_idx = threadIdx.x + (blockDim.x * blockIdx.x);
  int thread_strides = gridDim.x * blockDim.x;
  T *data_out = reinterpret_cast<T *>(dense_data_out_ptrs);
  unsigned int dense_id_offset = 0;
  for (int i = 0; i < num_dense; i++) {
    int64_t addr = dense_data_column_ptrs[i];
    T *data_in = reinterpret_cast<T *>(addr);
    for (int row_idx = start_idx; row_idx < batch_size * dense_dim_array[i];
         row_idx += thread_strides) {
      int sample_id = row_idx / dense_dim_array[i];
      int dense_id_sample = row_idx % dense_dim_array[i] + dense_id_offset;
      data_out[sample_id * label_dense_dim + dense_id_sample] = data_in[row_idx];
    }
    dense_id_offset += dense_dim_array[i];
  }
}

// caution that the input row_offset is always int32_t (According to the cudf definition)
// the output row_offset can be flexible: a template parameter
template <typename T>
void __global__ offset_kernel__(int64_t *row_offsets_src, int view_offset, int num_params,
                                int param_id, int slots_num, int64_t *row_len_dst, int batchsize) {
  extern __shared__ char smem_[];
  T *smem = reinterpret_cast<T *>(smem_);
  const int tid = threadIdx.x;
  const int gtid = tid + blockIdx.x * blockDim.x;
  const int lda = 33;
  for (int i = 0; i < slots_num; i += warpSize) {
    for (int j = 0; j < warpSize; j++) {
      int slot_id = i + j;
      if (slot_id < slots_num) {
        // multi_hot
        if (row_offsets_src[slot_id]) {
          int64_t addr = row_offsets_src[slot_id];
          int32_t *data = reinterpret_cast<int32_t *>(addr) + view_offset;
          if (gtid < batchsize) {
            smem[tid * lda + j] = static_cast<T>(data[gtid + 1] - data[gtid]);
          }

        }
        // one-hot
        else {
          if (gtid < batchsize) {
            smem[tid * lda + j] = static_cast<T>(1);
          }
        }
      }
    }
    __syncthreads();

    int out_row_idx = __shfl_sync(0xffffffff, gtid, 0);

    if ((__mylaneid() + i) < slots_num) {
      int warp_id = threadIdx.x / warpSize;
      int smem_row = warp_id * warpSize;

      for (int j = 0; j < warpSize; j++) {
        if ((j + out_row_idx) < batchsize) {
          int curr_out_row = j + out_row_idx;  // batch n
          // offset
          int slot_id = (__mylaneid() + i);
          int local_id = slot_id;

          uint32_t idx_to_buffers = curr_out_row * slots_num + local_id;
          int64_t addr = row_len_dst[param_id];
          T *data = reinterpret_cast<T *>(addr);
          data[idx_to_buffers + 1] = smem[(smem_row + j) * lda + __mylaneid()];
        }
      }
    }
  }
  // set first element to zero
  if (!gtid) {
    int64_t addr = row_len_dst[param_id];
    T *data = reinterpret_cast<T *>(addr);
    data[0] = 0;
  }
}
//! note that type of input row_offsets_src is int32_t, input slot_value_src is T
//! output row_offsets_dst & slot_value_out is T
template <typename ValueType, typename IndexType>
void __global__ value_kernel_without_shared_mem__(int64_t *row_offsets_src, int64_t *slot_value_src,
                                                  ValueType *dev_slot_offset_ptr, int view_offset,
                                                  int num_params, int param_id, int slots_num,
                                                  int64_t *row_offset_dst, int64_t *slot_value_out,
                                                  int batchsize) {
  // #samples = blockDim.x;
  const int sample_num = blockDim.x;
  const int total_dim = slots_num;
  const int tid = threadIdx.x;

  int sample_start = min(static_cast<int>(sample_num * blockIdx.x), static_cast<int>(batchsize));
  int sample_end =
      min(static_cast<int>(sample_num * (blockIdx.x + 1)), static_cast<int>(batchsize));
  for (int i = 0; i < total_dim; i++) {
    int64_t addr = row_offsets_src[i];
    int32_t *row_offsets_in = reinterpret_cast<int32_t *>(addr);
    addr = slot_value_src[i];
    ValueType *value_in = reinterpret_cast<ValueType *>(addr);
    int32_t row_start, row_end;
    int32_t proceeded_nnz = view_offset;

    if (row_offsets_in) {
      proceeded_nnz = row_offsets_in[view_offset];
      row_offsets_in += view_offset;
      row_start = row_offsets_in[sample_start] - proceeded_nnz;
      row_end = row_offsets_in[sample_end] - proceeded_nnz;
    } else {
      row_start = sample_start;
      row_end = sample_end;
    }
    value_in += proceeded_nnz;
    int buffer_id = param_id;
    addr = row_offset_dst[buffer_id];
    IndexType *row_offset_write = reinterpret_cast<IndexType *>(addr);

    // load to reg and store to global memory directly
    for (int row_idx = tid + row_start; row_idx < row_end; row_idx += blockDim.x) {
      int sample_id;
      if (row_offsets_in) {
        sample_id =
            static_cast<int>(upper_bound(row_offsets_in + sample_start, row_offsets_in + sample_end,
                                         row_idx + proceeded_nnz) -
                             row_offsets_in - 1);
      } else {
        sample_id = row_idx;
      }
      unsigned int idx_to_buffers = sample_id * slots_num + i;

      IndexType dst_offset = static_cast<IndexType>(0);
      // m-hot
      if (row_offsets_in) {
        dst_offset = row_offset_write[idx_to_buffers];
        int slot_id = static_cast<int32_t>(row_idx) + proceeded_nnz - row_offsets_in[sample_id];
        dst_offset += slot_id;
      } else {  // s-hot
        dst_offset = row_offset_write[idx_to_buffers];
      }

      addr = slot_value_out[buffer_id];
      ValueType *value_write = reinterpret_cast<ValueType *>(addr);
      value_write[dst_offset] = value_in[row_idx] + dev_slot_offset_ptr[i];
    }
  }
}

/**
 * Interleave dense (continuous) data parquet columns and write to linear buffer
 * @param dense_column_data_ptr vector of device pointers to Parquet columns
 * @param num_dense number of dense values (lable_num + dense_num); the dense_num_array size
 * @param label_dense_dim number of dense values (lable_dim + dense_dim);label_dense_num =
 * sum(dense_dim_array_ptr)
 * @param dense_dim_array_device_ptr pointer to GPU memory for dense_dim_array_ptr
 * @param batch_size batch size to load (current_batch_size)
 * @param batch_start sample start to load for local gpus
 * @param batch_end sample end to to load for local gpus
 * @param dense_data_buffers buffers to write output
 * @param dev_ptr_staging pointer to pinned memory for copying pointer address from h2d
 * @param rmm_buffers Queue to hold reference to RMM allocations
 * @param mr Device memory resource for RMM allocations
 * @param task_stream Stream to allocate memory and launch kerenels
 */
template <typename T>
void convert_parquet_dense_columns(std::vector<T *> &dense_column_data_ptr, const int num_dense,
                                   int64_t *dense_dim_array_device_ptr, const int label_dense_dim,
                                   int batch_size, int batch_start, int batch_end,
                                   T *dense_data_buffers, int64_t *dev_ptr_staging,
                                   std::deque<rmm::device_buffer> &rmm_buffers,
                                   rmm::mr::device_memory_resource *mr, cudaStream_t task_stream) {
  int samples_to_interleaved = batch_size;
  if (!samples_to_interleaved) return;
  size_t size_of_col_ptrs = dense_column_data_ptr.size() * sizeof(T *);
  std::memcpy(dev_ptr_staging, dense_column_data_ptr.data(), size_of_col_ptrs);

  rmm_buffers.emplace_back(size_of_col_ptrs, task_stream, mr);
  rmm::device_buffer &dev_in_column_ptr = rmm_buffers.back();
  // fill empty sample dense features
  HCTR_LIB_THROW(cudaMemcpyAsync(dev_in_column_ptr.data(), dev_ptr_staging, size_of_col_ptrs,
                                 cudaMemcpyHostToDevice, task_stream));
  // assuming 48KB smem/SM
  // 32x32 tile per warp -> 4096 bytes/warp
  // 12 warps -> 384 threads/block
  // size_t smem_size = 48 * 1024 * 1024;

  HCTR_LIB_THROW(cudaMemsetAsync(
      dense_data_buffers, 0, sizeof(T) * label_dense_dim * (batch_end - batch_start), task_stream));
  // all dense columns' type is scalar
  if (num_dense == label_dense_dim) {
    dim3 block(256, 1, 1);
    dim3 grid((samples_to_interleaved - 1) / block.x + 1, 1, 1);
    dense_data_converter_kernel_scalar_<T>
        <<<grid, block, 0, task_stream>>>((int64_t *)dev_in_column_ptr.data(), label_dense_dim,
                                          samples_to_interleaved, dense_data_buffers);
  } else {
    dim3 block(256, 1, 1);
    dim3 grid((samples_to_interleaved - 1) / block.x + 1, 1, 1);
    dense_data_converter_kernel__<T><<<grid, block, 0, task_stream>>>(
        (int64_t *)dev_in_column_ptr.data(), num_dense, dense_dim_array_device_ptr, label_dense_dim,
        samples_to_interleaved, dense_data_buffers);
  }
  HCTR_LIB_THROW(cudaGetLastError());
  return;
}

/**
 * Interleave categoricals (slot) data parquet columns and write to csr buffers
 * @param cat_column_data_ptr [input] vector of device pointers to Parquet columns
 * @param cat_column_row_offset_ptr [input] vector of device pointers to Parquet row_offset (m-hot)
 * @param view_offset starting index of current batch in parquet column
 * @param num_params number of Embedding params
 * @param param_id param idx for current param
 * @param max_nnz max_nnz for all slots
 * @param num_slots number of slots in current param
 * @param batch_size batch size to load
 * @param pid pid of node
 * @param resource_manager ResourceManager handle for session
 * @param csr_value_buffers [output] vector of device buffers to write csr values
 * @param csr_row_offset_buffers [output] vector of device buffers to write csr row offset values
 * @param dev_ptr_staging pointer to pinned memory for copying pointer address from h2d
 * @param rmm_buffers Queue to hold reference to RMM allocations
 * @param mr Device memory resource for RMM allocations
 * @param task_stream Stream to allocate memory and launch kerenels
 */

template <typename T, typename IndexType>
size_t convert_parquet_cat_columns(
    std::vector<T *> &cat_column_data_ptr, std::vector<int32_t *> &cat_column_row_offset_ptr,
    int view_offset, int num_params, int param_id, int max_nnz, int num_slots, int batch_size,
    int pid, const std::shared_ptr<ResourceManager> resource_manager,
    std::vector<void *> &csr_value_buffers, std::vector<void *> &csr_row_offset_buffers,
    int64_t *dev_ptr_staging, T *dev_slot_offset_ptr, std::deque<rmm::device_buffer> &rmm_buffers,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream) {
  size_t pinned_staging_elements_used = 0;
  size_t size_of_col_ptrs = cat_column_data_ptr.size() * sizeof(int64_t *);

  // prepare for value input
  std::memcpy(dev_ptr_staging, cat_column_data_ptr.data(), size_of_col_ptrs);
  pinned_staging_elements_used += cat_column_data_ptr.size();

  rmm_buffers.emplace_back(size_of_col_ptrs, task_stream, mr);
  rmm::device_buffer &dev_in_column_ptr = rmm_buffers.back();
  HCTR_LIB_THROW(cudaMemcpyAsync(dev_in_column_ptr.data(), dev_ptr_staging, size_of_col_ptrs,
                                 cudaMemcpyHostToDevice, task_stream));

  int64_t *pinned_csr_offset_in_buffer =
      reinterpret_cast<int64_t *>((size_t)(dev_ptr_staging) + size_of_col_ptrs);

  // prepare for row_offset input
  std::memcpy(pinned_csr_offset_in_buffer, cat_column_row_offset_ptr.data(), size_of_col_ptrs);
  pinned_staging_elements_used += cat_column_row_offset_ptr.size();

  rmm_buffers.emplace_back(size_of_col_ptrs, task_stream, mr);
  rmm::device_buffer &dev_csr_offset_in_buffer = rmm_buffers.back();
  HCTR_LIB_THROW(cudaMemcpyAsync(dev_csr_offset_in_buffer.data(), pinned_csr_offset_in_buffer,
                                 size_of_col_ptrs, cudaMemcpyHostToDevice, task_stream));

  size_t size_of_csr_pointers = num_params * sizeof(int64_t);

  int64_t *pinned_csr_val_out_buffer =
      reinterpret_cast<int64_t *>((size_t)(dev_ptr_staging) + 2 * size_of_col_ptrs);
  for (int i = 0; i < num_params; i++) {
    pinned_csr_val_out_buffer[i] = (int64_t)csr_value_buffers[i];
  }
  pinned_staging_elements_used += num_params;

  rmm_buffers.emplace_back(size_of_csr_pointers, task_stream, mr);
  rmm::device_buffer &dev_csr_value_ptr = rmm_buffers.back();
  HCTR_LIB_THROW(cudaMemcpyAsync(dev_csr_value_ptr.data(), pinned_csr_val_out_buffer,
                                 size_of_csr_pointers, cudaMemcpyHostToDevice, task_stream));

  int64_t *pinned_csr_row_offset_buffer = reinterpret_cast<int64_t *>(
      (size_t)(dev_ptr_staging) + 2 * size_of_col_ptrs + size_of_csr_pointers);
  for (int i = 0; i < num_params; i++) {
    pinned_csr_row_offset_buffer[i] = (int64_t)csr_row_offset_buffers[i];
  }
  pinned_staging_elements_used += num_params;

  rmm_buffers.emplace_back(size_of_csr_pointers, task_stream, mr);
  rmm::device_buffer &dev_csr_row_offset_ptr = rmm_buffers.back();
  HCTR_LIB_THROW(cudaMemcpyAsync(dev_csr_row_offset_ptr.data(), pinned_csr_row_offset_buffer,
                                 size_of_csr_pointers, cudaMemcpyHostToDevice, task_stream));
  {
    int block_size = (sizeof(IndexType) == 8) ? 128 : 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((batch_size - 1) / block.x + 1, 1, 1);
    size_t smem_size = (block_size / 32) * sizeof(IndexType) * 32 * 33;
    size_t max_smem_size = 48 * 1024;

    if (smem_size > max_smem_size)
      HCTR_OWN_THROW(Error_t::OutOfMemory, "Parquet Converter: Not enough shared memory availble");

    offset_kernel__<IndexType><<<grid, block, smem_size, task_stream>>>(
        reinterpret_cast<int64_t *>(dev_csr_offset_in_buffer.data()), view_offset, num_params,
        param_id, num_slots, reinterpret_cast<int64_t *>(dev_csr_row_offset_ptr.data()),
        batch_size);
    int buffer_id = param_id;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int64_t prefix_sum_items = num_slots * batch_size + 1;
    HCTR_LIB_THROW(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        reinterpret_cast<IndexType *>(pinned_csr_row_offset_buffer[buffer_id]),
        reinterpret_cast<IndexType *>(pinned_csr_row_offset_buffer[buffer_id]), prefix_sum_items,
        task_stream));
    rmm_buffers.emplace_back(temp_storage_bytes, task_stream, mr);
    rmm::device_buffer &cub_tmp_storage = rmm_buffers.back();
    HCTR_LIB_THROW(cub::DeviceScan::InclusiveSum(
        cub_tmp_storage.data(), temp_storage_bytes,
        reinterpret_cast<IndexType *>(pinned_csr_row_offset_buffer[buffer_id]),
        reinterpret_cast<IndexType *>(pinned_csr_row_offset_buffer[buffer_id]), prefix_sum_items,
        task_stream));

    int num_keys = 512;

    int samples_per_block = (num_keys - 1) / max_nnz + 1;
    int block_per_grid = (batch_size - 1) / samples_per_block + 1;

    value_kernel_without_shared_mem__<T, IndexType>
        <<<block_per_grid, samples_per_block, 0, task_stream>>>(
            reinterpret_cast<int64_t *>(dev_csr_offset_in_buffer.data()),
            reinterpret_cast<int64_t *>(dev_in_column_ptr.data()), dev_slot_offset_ptr, view_offset,
            num_params, param_id, num_slots,
            reinterpret_cast<int64_t *>(dev_csr_row_offset_ptr.data()),
            reinterpret_cast<int64_t *>(dev_csr_value_ptr.data()), batch_size);

    HCTR_LIB_THROW(cudaGetLastError());
  }

  return pinned_staging_elements_used;
}

// init function instances here
template void convert_parquet_dense_columns<float>(
    std::vector<float *> &dense_column_data_ptr, const int num_dense, int64_t *dense_dim_array_ptr,
    const int label_dense_dim, int batch_size, int batch_start, int batch_end,
    float *dense_data_buffers, int64_t *dev_ptr_staging,
    std::deque<rmm::device_buffer> &rmm_buffers, rmm::mr::device_memory_resource *mr,
    cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<long long int, int32_t>(
    std::vector<long long int *> &cat_column_data_ptr,
    std::vector<int32_t *> &cat_column_row_offset_ptr, int view_offset, int num_params,
    int param_id, int nnz, int num_slots, int batch_size, int pid,
    const std::shared_ptr<ResourceManager> resource_manager, std::vector<void *> &csr_value_buffers,
    std::vector<void *> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    long long int *slot_offset, std::deque<rmm::device_buffer> &rmm_buffers,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<unsigned int, int32_t>(
    std::vector<unsigned int *> &cat_column_data_ptr,
    std::vector<int32_t *> &cat_column_row_offset_ptr, int view_offset, int num_params,
    int param_id, int nnz, int num_slots, int batch_size, int pid,
    const std::shared_ptr<ResourceManager> resource_manager, std::vector<void *> &csr_value_buffers,
    std::vector<void *> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    unsigned int *slot_offset, std::deque<rmm::device_buffer> &rmm_buffers,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<long long int, int64_t>(
    std::vector<long long int *> &cat_column_data_ptr,
    std::vector<int32_t *> &cat_column_row_offset_ptr, int view_offset, int num_params,
    int param_id, int nnz, int num_slots, int batch_size, int pid,
    const std::shared_ptr<ResourceManager> resource_manager, std::vector<void *> &csr_value_buffers,
    std::vector<void *> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    long long int *slot_offset, std::deque<rmm::device_buffer> &rmm_buffers,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream);

template size_t convert_parquet_cat_columns<unsigned int, int64_t>(
    std::vector<unsigned int *> &cat_column_data_ptr,
    std::vector<int32_t *> &cat_column_row_offset_ptr, int view_offset, int num_params,
    int param_id, int nnz, int num_slots, int batch_size, int pid,
    const std::shared_ptr<ResourceManager> resource_manager, std::vector<void *> &csr_value_buffers,
    std::vector<void *> &csr_row_offset_buffers, int64_t *dev_ptr_staging,
    unsigned int *slot_offset, std::deque<rmm::device_buffer> &rmm_buffers,
    rmm::mr::device_memory_resource *mr, cudaStream_t task_stream);
}  // namespace HugeCTR
