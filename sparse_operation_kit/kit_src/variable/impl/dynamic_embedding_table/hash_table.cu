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

#include "hash_table.hpp"

namespace {

template <typename value_type>
struct ReplaceOp {
  // constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_or_assign_kernel(Table *table, const typename Table::key_type *const keys,
                                        const typename Table::mapped_type *const vals, size_t len) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    kv.first = keys[i];
    kv.second = vals[i];
    auto it = table->insert_or_assign(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table>
__global__ void lookup_kernel(Table *table, const typename Table::key_type *const keys,
                              typename Table::mapped_type *const vals, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    assert(it != table->end() && "error: can't find key");
    vals[i] = it->second;
  }
}

template <typename Table>
__global__ void lookup_or_insert_kernel(Table *table, const typename Table::key_type *const keys,
                                        typename Table::mapped_type *const vals, size_t len,
                                        size_t *d_counter) {
  ReplaceOp<typename Table::mapped_type> op;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->lookup_or_insert(keys[i], op, d_counter);
    assert(it != table->end() && "error: get_insert fails: table is full");
    vals[i] = it->second;
  }
}

template <typename Table, typename KeyType>
__global__ void size_kernel(const Table *table, const size_t hash_capacity, size_t *container_size,
                            KeyType unused_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  /* Whether the bucket mapping to the current thread is empty? do nothing : Atomically add to
   * counter */
  if (i < hash_capacity) {
    typename Table::value_type val = load_pair_vectorized(table->data() + i);
    if (val.first != unused_key) {
      atomicAdd(&block_acc, 1);
    }
  }
  __syncthreads();

  /* Atomically reduce block counter to global conuter */
  if (threadIdx.x == 0) {
    atomicAdd(container_size, block_acc);
  }
}

}  // namespace

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity, size_t count) : capacity_(capacity) {
  // Allocate device-side counter and copy user input to it
  container_ = new concurrent_unordered_map<KeyType, ValType, std::numeric_limits<KeyType>::max()>(
      static_cast<size_t>(capacity / LOAD_FACTOR), std::numeric_limits<ValType>::max());
  CUDA_RT_CALL(cudaMalloc((void **)&d_counter_, sizeof(size_t)));
  CUDA_RT_CALL(cudaMalloc((void **)&d_container_size_, sizeof(size_t)));
  CUDA_RT_CALL(cudaMemcpy(d_counter_, &count, sizeof(size_t), cudaMemcpyHostToDevice));
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  try {
    delete container_;
    // De-allocate device-side counter
    CUDA_RT_CALL(cudaFree(d_counter_));
    CUDA_RT_CALL(cudaFree(d_container_size_));
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert_or_assign(const KeyType *d_keys, const ValType *d_vals,
                                                   size_t len, cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  insert_or_assign_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::lookup_or_insert(const KeyType *d_keys, ValType *d_vals,
                                                   size_t len, cudaStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  lookup_or_insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len,
                                                                 d_counter_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::lookup(const KeyType *d_keys, ValType *d_vals, size_t len,
                                         cudaStream_t stream) const {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  lookup_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, d_keys, d_vals, len);
}

template <typename KeyType, typename ValType>
size_t HashTable<KeyType, ValType>::get_size(cudaStream_t stream) const {
  /* size variable on Host and device, total capacity of the hashtable */
  size_t container_size;

  const size_t hash_capacity = container_->size();

  /* grid_size and allocating/initializing variable on dev, launching kernel*/
  const int grid_size = (hash_capacity - 1) / BLOCK_SIZE_ + 1;

  CUDA_RT_CALL(cudaMemsetAsync(d_container_size_, 0, sizeof(size_t), stream));
  size_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(container_, hash_capacity, d_container_size_,
                                                     empty_key);
  CUDA_RT_CALL(cudaMemcpyAsync(&container_size, d_container_size_, sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
  // CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  return container_size;
}

template <typename KeyType, typename ValType>
size_t HashTable<KeyType, ValType>::get_capacity() const {
  return container_->size();
}

template class HashTable<int64_t, size_t>;
template class HashTable<int64_t, int64_t>;
