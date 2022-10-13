/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

#ifndef NUMA_NODE_MAP
#define NUMA_NODE_MAP
std::unordered_map<int, int> CudaCPUDeviceContext::device_id_to_numa_node_;
#endif

template <typename TIN, typename TOUT>
void convert_array_on_device(TOUT *out, const TIN *in, size_t num_elements,
                             const cudaStream_t &stream) {
  if (num_elements > 0) {
    convert_array<<<(num_elements - 1) / 1024 + 1, 1024, 0, stream>>>(out, in, num_elements);
  }
}

template void convert_array_on_device<long long, int>(int *, const long long *, size_t,
                                                      const cudaStream_t &);
template void convert_array_on_device<unsigned int, int>(int *, const unsigned int *, size_t,
                                                         const cudaStream_t &);
template void convert_array_on_device<float, float>(float *, const float *, size_t,
                                                    const cudaStream_t &);
template void convert_array_on_device<float, __half>(__half *, const float *, size_t,
                                                     const cudaStream_t &);
template void convert_array_on_device<__half, float>(float *, const __half *, size_t,
                                                     const cudaStream_t &);

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *rowoffset,
                               const TypeKey *emmbedding_offsets, int num_tables,
                               int num_rowoffsets, const cudaStream_t &stream) {
  constexpr size_t block_size = 256;
  size_t grid_size = (num_rowoffsets - 1) / block_size + 1;
  unique_key_kernels::data_to_unique_categories_kernel<<<grid_size, block_size, 0, stream>>>(
      value, rowoffset, emmbedding_offsets, num_tables, num_rowoffsets);
}

template void data_to_unique_categories<long long>(long long *, const long long *,
                                                   const long long *, int, int,
                                                   const cudaStream_t &);

template void data_to_unique_categories<unsigned int>(unsigned int *, const unsigned int *,
                                                      const unsigned int *, int, int,
                                                      const cudaStream_t &);

template <typename TypeKey>
void data_to_unique_categories(TypeKey *value, const TypeKey *emmbedding_offsets, int num_tables,
                               int nnz, const cudaStream_t &stream) {
  constexpr size_t block_size = 256;
  size_t grid_size = std::min(4096ul, (nnz - 1) / block_size + 1);
  if (num_tables % 2 == 0 && sizeof(TypeKey) == 4) {
    unique_key_kernels::
        data_to_unique_categories_align2_kernel<<<grid_size, block_size, 0, stream>>>(
            value, emmbedding_offsets, num_tables / 2, nnz / 2);
  } else {
    unique_key_kernels::data_to_unique_categories_kernel<<<grid_size, block_size, 0, stream>>>(
        value, emmbedding_offsets, num_tables, nnz);
  }
}

template void data_to_unique_categories<long long>(long long *, const long long *, int, int,
                                                   const cudaStream_t &);

template void data_to_unique_categories<unsigned int>(unsigned int *, const unsigned int *, int,
                                                      int, const cudaStream_t &);

template <typename T>
__global__ void inc_var_cuda(T *x) {
  if (blockIdx.x == 0 and threadIdx.x == 0) {
    (*x)++;
  }
}

template <typename T>
void inc_var(volatile T *x, cudaStream_t stream) {
  inc_var_cuda<<<1, 32, 0, stream>>>(x);
}

template void inc_var<size_t>(volatile size_t *x, cudaStream_t stream);

__global__ void calc_embedding_offset_cuda(size_t *d_embedding_offset, const int *d_row_ptrs,
                                           const size_t *d_row_ptrs_offset,
                                           const size_t *d_slot_num_for_tables, size_t num_tables,
                                           size_t batch_size, bool sample_first) {
  size_t thread_num = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_num >= num_tables * batch_size) return;
  size_t table_num;
  size_t batch_num;
  if (sample_first) {
    batch_num = thread_num / num_tables;
    table_num = thread_num % num_tables;
  } else {
    table_num = thread_num / batch_size;
    batch_num = thread_num % batch_size;
  }
  const int *const d_row_ptrs_per_table = d_row_ptrs + d_row_ptrs_offset[table_num];
  const size_t num_of_feature =
      d_row_ptrs_per_table[(batch_num + 1) * d_slot_num_for_tables[table_num]] -
      d_row_ptrs_per_table[batch_num * d_slot_num_for_tables[table_num]];
  if (sample_first) {
    d_embedding_offset[batch_num * num_tables + table_num + 1] =
        d_embedding_offset[batch_num * num_tables + table_num] + num_of_feature;
  } else {
    d_embedding_offset[table_num * batch_size + batch_num + 1] =
        d_embedding_offset[table_num * batch_size + batch_num] + num_of_feature;
  }
}

void calc_embedding_offset(size_t *d_embedding_offset, const int *d_row_ptrs,
                           const size_t *d_row_ptrs_offset, const size_t *d_slot_num_for_tables,
                           size_t num_tables, size_t batch_size, bool sample_first,
                           cudaStream_t stream) {
  calc_embedding_offset_cuda<<<(num_tables * batch_size - 1) / 1024 + 1, 1024, 0, stream>>>(
      d_embedding_offset, d_row_ptrs, d_row_ptrs_offset, d_slot_num_for_tables, num_tables,
      batch_size, sample_first);
}

template <typename TypeHashKey>
__global__ void convert_keys_to_table_first_cuda(TypeHashKey *d_out, const TypeHashKey *d_in,
                                                 size_t *d_embedding_offset_table_first,
                                                 size_t *d_embedding_offset_sample_first,
                                                 size_t num_tables, size_t batch_size) {
  size_t thread_num = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_num >= num_tables * batch_size) return;
  size_t batch_num = thread_num / num_tables;
  size_t table_num = thread_num % num_tables;

  const size_t num_keys = d_embedding_offset_sample_first[batch_num * num_tables + table_num + 1] -
                          d_embedding_offset_sample_first[batch_num * num_tables + table_num];

  for (size_t k = 0; k < num_keys; k++) {
    d_out[d_embedding_offset_table_first[table_num * batch_size + batch_num] + k] =
        d_in[d_embedding_offset_sample_first[batch_num * num_tables + table_num] + k];
  }
}

template void convert_keys_to_table_first<long long>(long long *d_out, const long long *d_in,
                                                     size_t *d_embedding_offset_table_first,
                                                     size_t *d_embedding_offset_sample_first,
                                                     size_t num_tables, size_t batch_size,
                                                     cudaStream_t stream);

template void convert_keys_to_table_first<unsigned int>(unsigned int *d_out,
                                                        const unsigned int *d_in,
                                                        size_t *d_embedding_offset_table_first,
                                                        size_t *d_embedding_offset_sample_first,
                                                        size_t num_tables, size_t batch_size,
                                                        cudaStream_t stream);

template <typename TypeHashKey>
void convert_keys_to_table_first(TypeHashKey *d_out, const TypeHashKey *d_in,
                                 size_t *d_embedding_offset_table_first,
                                 size_t *d_embedding_offset_sample_first, size_t num_tables,
                                 size_t batch_size, cudaStream_t stream) {
  convert_keys_to_table_first_cuda<<<(num_tables * batch_size - 1) / 1024 + 1, 1024, 0, stream>>>(
      d_out, d_in, d_embedding_offset_table_first, d_embedding_offset_sample_first, num_tables,
      batch_size);
}

}  // namespace HugeCTR
