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

#include <core23/cuda_primitives.cuh>
#include <hps/unique_op/unique_op.hpp>

namespace HugeCTR {
namespace unique_op {

template <typename KeyType, typename CounterType, typename TableEntry>
__global__ void init_kernel(KeyType* keys, TableEntry* table, CounterType* vals,
                            CounterType* counter, const size_t capacity, const KeyType empty_key,
                            const CounterType empty_val, const CounterType init_counter_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    // Simply store every element a unused <K, V> pair
    keys[idx] = empty_key;
    vals[idx] = empty_val;
    table[idx].key = empty_key;
    KeyEntry<KeyType> emptyKey{empty_key};
    table[idx].value.write(0, emptyKey);
  }
  if (idx == 0) {
    counter[idx] = init_counter_val;
  }
}

template <typename KeyType, typename TableEntry, typename CounterType, typename hasher>
__global__ void unique_get_insert_dump_kernel(const KeyType* d_key, CounterType* d_val,
                                              TableEntry* d_table, const size_t len, KeyType* keys,
                                              CounterType* vals, const size_t capacity,
                                              CounterType* d_global_counter, KeyType* d_unique_key,
                                              const KeyType empty_key,
                                              const CounterType empty_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    KeyType target_key = d_key[idx];
    KeyEntry<KeyType> unique_out = {empty_key};
    bool is_unique = true;
    auto r_idx = core23 ::get_insert_dump<KeyEntry<KeyType>, TableEntry, CounterType, hasher>(
        {target_key}, d_table, d_global_counter, unique_out, capacity, {empty_key}, is_unique);
    if (is_unique) {
      d_unique_key[r_idx - 1] = unique_out.key;
    }
    d_val[idx] = r_idx - 1;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::unique_op(
    const size_t capacity, const CounterType init_counter_val)
    : capacity_(capacity), init_counter_val_(init_counter_val) {
  // Check parameter
  if (capacity_ == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Invalid value for unique_op capacity");
    return;
  }

  // Get the current CUDA dev
  HCTR_LIB_THROW(cudaGetDevice(&dev_));

  // Allocate keys and vals buffer
  HCTR_LIB_THROW(cudaMalloc((void**)&keys_, sizeof(KeyType) * capacity_));
  HCTR_LIB_THROW(cudaMalloc((void**)&vals_, sizeof(CounterType) * capacity_));
  HCTR_LIB_THROW(cudaMalloc((void**)&table_, sizeof(TableEntry<KeyType>) * capacity_));

  // Allocate device-side counter
  HCTR_LIB_THROW(cudaMalloc((void**)&counter_, sizeof(CounterType)));

  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType, TableEntry<KeyType>>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
          keys_, table_, vals_, counter_, capacity_, empty_key, empty_val, init_counter_val_);

  // Wait for initialization to finish
  HCTR_LIB_THROW(cudaStreamSynchronize(0));
  HCTR_LIB_THROW(cudaGetLastError());
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::~unique_op() noexcept(false) {
  // Device Restorer
  CudaDeviceContext dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Free keys and vals
  HCTR_LIB_THROW(cudaFree(keys_));
  HCTR_LIB_THROW(cudaFree(vals_));

  // Free device-side counter
  HCTR_LIB_THROW(cudaFree(counter_));
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
size_t unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::get_capacity() const {
  return capacity_;
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::unique(
    const KeyType* d_key, const size_t len, CounterType* d_output_index, KeyType* d_unique_key,
    size_t* d_output_counter, cudaStream_t stream) {
  // Device Restorer
  CudaDeviceContext dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Set the d_output_counter to 0
  HCTR_LIB_THROW(cudaMemsetAsync(d_output_counter, 0, sizeof(size_t), stream));

  if (len == 0) {
    return;
  }

  // Launch get_insert kernel to do unique
  unique_get_insert_dump_kernel<KeyType, TableEntry<KeyType>, CounterType, hasher>
      <<<(len - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0, stream>>>(d_key, d_output_index, table_, len,
                                                                keys_, vals_, capacity_, counter_,
                                                                d_unique_key, empty_key, empty_val);

  HCTR_LIB_THROW(
      cudaMemcpyAsync(d_output_counter, counter_, sizeof(size_t), cudaMemcpyHostToHost, stream));

  HCTR_LIB_THROW(cudaGetLastError());
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::clear(cudaStream_t stream) {
  // Device Restorer
  CudaDeviceContext dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType, TableEntry<KeyType>>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, stream>>>(
          keys_, table_, vals_, counter_, capacity_, empty_key, empty_val, init_counter_val_);

  HCTR_LIB_THROW(cudaGetLastError());
}

template class unique_op<int32_t, uint64_t, std::numeric_limits<int32_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<uint32_t, uint64_t, std::numeric_limits<uint32_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<int64_t, uint64_t, std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<uint64_t, uint64_t, std::numeric_limits<uint64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<long long, uint64_t, std::numeric_limits<long long>::max(),
                         std::numeric_limits<uint64_t>::max()>;
}  // namespace unique_op
}  // namespace HugeCTR
