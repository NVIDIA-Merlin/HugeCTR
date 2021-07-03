
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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
namespace HugeCTR {

namespace {

// memset liner data to the buffer
template <typename Type>
__global__ void memset_liner_kernel(Type *data, const Type start_value, const Type stride_value,
                                    size_t n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = start_value + gid * stride_value;
  }
}

// memset constant data to the buffer
template <typename Type>
__global__ void memset_const_kernel(Type *data, const Type value, long long n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = value;
  }
}

// get hash_value by value_index from hash_table_value matrix
template <typename TypeValueIndex>
__global__ void get_hash_value_kernel(long long count, int embedding_vec_size,
                                      const TypeValueIndex *value_index,
                                      const float *hash_table_value, float *value_retrieved) {
  int tid = threadIdx.x;
  size_t bid = blockIdx.x;

  if (bid < count && tid < embedding_vec_size) {
    size_t index = value_index[bid];  // row number in the hash_table_value matrix
    value_retrieved[bid * embedding_vec_size + tid] =
        hash_table_value[index * embedding_vec_size + tid];
  }
}

}  // namespace

template <typename Type>
void SparseEmbeddingFunctors::memset_liner(Type *data, Type start_value, Type stride_value,
                                           size_t n, cudaStream_t stream) const {
  const size_t block_size = 256;
  const size_t grid_size = (n + block_size - 1) / block_size;

  memset_liner_kernel<<<grid_size, block_size, 0, stream>>>(data, start_value, stride_value, n);
}

void SparseEmbeddingFunctors::memset_const(size_t *data, size_t value, size_t n,
                                           cudaStream_t stream) const {
  const size_t block_size = 256;
  const size_t grid_size = (n + block_size - 1) / block_size;

  memset_const_kernel<<<grid_size, block_size, 0, stream>>>(data, value, n);
}

void SparseEmbeddingFunctors::get_hash_value(size_t count, size_t embedding_vec_size,
                                             const size_t *value_index,
                                             const float *hash_table_value, float *value_retrieved,
                                             cudaStream_t stream) const {
  const size_t block_size = embedding_vec_size;
  const size_t grid_size = count;

  get_hash_value_kernel<<<grid_size, block_size, 0, stream>>>(
      count, embedding_vec_size, value_index, hash_table_value, value_retrieved);
}

template void SparseEmbeddingFunctors::memset_liner<unsigned int>(unsigned int *data,
                                                                  unsigned int start_value,
                                                                  unsigned int stride_value,
                                                                  size_t n,
                                                                  cudaStream_t stream) const;

template void SparseEmbeddingFunctors::memset_liner<long long>(long long *data,
                                                               long long start_value,
                                                               long long stride_value, size_t n,
                                                               cudaStream_t stream) const;

template void SparseEmbeddingFunctors::memset_liner<size_t>(size_t *data, size_t start_value,
                                                            size_t stride_value, size_t n,
                                                            cudaStream_t stream) const;

}  // namespace HugeCTR