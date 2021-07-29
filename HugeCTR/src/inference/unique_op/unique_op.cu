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

#include <inference/unique_op/unique_op.hpp>

// Overload CUDA atomic for other 64bit unsinged/signed integer type
__forceinline__ __device__ long atomicAdd(long* address, long val) {
  return (long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long* address, long long val) {
  return (long long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return (unsigned long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long atomicCAS(long* address, long compare, long val) {
  return (long)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                         (unsigned long long)val);
}

__forceinline__ __device__ long long atomicCAS(long long* address, long long compare,
                                               long long val) {
  return (long long)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicCAS(unsigned long* address, unsigned long compare,
                                                   unsigned long val) {
  return (unsigned long)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                                  (unsigned long long)val);
}

namespace HugeCTR {
namespace unique_op {

template <typename KeyType, typename CounterType>
__global__ void init_kernel(KeyType* keys, CounterType* vals, CounterType* counter,
                            const size_t capacity, const KeyType empty_key,
                            const CounterType empty_val, const CounterType init_counter_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    // Simply store every element a unused <K, V> pair
    keys[idx] = empty_key;
    vals[idx] = empty_val;
  }
  if (idx == 0) {
    counter[idx] = init_counter_val;
  }
}

template <typename KeyType, typename CounterType>
__global__ void dump_kernel(KeyType* d_key, const KeyType* keys, const CounterType* vals,
                            const size_t offset, const size_t search_length, size_t* d_dump_counter,
                            const KeyType empty_key) {
  /* Per block accumulator */
  __shared__ size_t block_acc;

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Initialize */
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  KeyType read_key;
  CounterType read_val;
  bool valid_slot = false;
  // Each thread gather the key and value from slot assigned to them.
  if (idx < search_length) {
    read_key = keys[offset + idx];
    if (read_key != empty_key) {
      valid_slot = true;
      atomicAdd(&block_acc, 1);
      read_val = vals[offset + idx];
    }
  }
  __syncthreads();

  // Each block accumulate the dump count to global counter
  if (threadIdx.x == 0) {
    atomicAdd(d_dump_counter, block_acc);
  }

  // Each thread store one slot's data back to global memory, d_dump_counter is how many slots in
  // total dumped.
  if (valid_slot) {
    d_key[read_val] = read_key;
  }
}

template <typename KeyType, typename CounterType, typename hasher>
__global__ void get_insert_kernel(const KeyType* d_key, CounterType* d_val, const size_t len,
                                  KeyType* keys, CounterType* vals, const size_t capacity,
                                  CounterType* d_global_counter, const KeyType empty_key,
                                  const CounterType empty_val) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    KeyType target_key = d_key[idx];
    size_t hash_index = hasher::hash(target_key) % capacity;
    size_t counter = 0;
    while (true) {
      // Have searched all the slot in the hashtable, but all slots in the hashtable are occupied by
      // other keys
      if (counter >= capacity) {
        assert(false && "error: unique op fails: hashtable is full");
      }
      // Try to set the key for the current slot to target key
      const KeyType old_key = atomicCAS(keys + hash_index, empty_key, target_key);
      volatile CounterType& target_val_pos = vals[hash_index];
      if (empty_key == old_key) {
        CounterType result_val;
        result_val = atomicAdd(d_global_counter, 1);
        d_val[idx] = result_val;
        target_val_pos = result_val;
        break;
      } else if (target_key == old_key) {
        while (target_val_pos == empty_val)
          ;
        d_val[idx] = target_val_pos;
        break;
      }
      counter++;
      hash_index = (hash_index + 1) % capacity;
    }
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
    CK_THROW_(Error_t::WrongInput, "Invalid value for unique_op capacity");
    return;
  }

  // Get the current CUDA dev
  CK_CUDA_THROW_(cudaGetDevice(&dev_));

  // Allocate keys and vals buffer
  CK_CUDA_THROW_(cudaMalloc((void**)&keys_, sizeof(KeyType) * capacity_));
  CK_CUDA_THROW_(cudaMalloc((void**)&vals_, sizeof(CounterType) * capacity_));

  // Allocate device-side counter
  CK_CUDA_THROW_(cudaMalloc((void**)&counter_, sizeof(CounterType)));

  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType><<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
      keys_, vals_, counter_, capacity_, empty_key, empty_val, init_counter_val_);

  // Wait for initialization to finish
  CK_CUDA_THROW_(cudaStreamSynchronize(0));
  CK_CUDA_THROW_(cudaGetLastError());
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::~unique_op() noexcept(false) {
  // Device Restorer
  CudaDeviceContext dev_restorer;
  // Set device
  CK_CUDA_THROW_(cudaSetDevice(dev_));

  // Free keys and vals
  CK_CUDA_THROW_(cudaFree(keys_));
  CK_CUDA_THROW_(cudaFree(vals_));

  // Free device-side counter
  CK_CUDA_THROW_(cudaFree(counter_));
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
  // Set to the device of this op
  CK_CUDA_THROW_(cudaSetDevice(dev_));

  // Set the d_output_counter to 0
  CK_CUDA_THROW_(cudaMemsetAsync(d_output_counter, 0, sizeof(size_t), stream));

  if (len == 0) {
    return;
  }

  // Launch get_insert kernel to do unique
  get_insert_kernel<KeyType, CounterType, hasher>
      <<<(len - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0, stream>>>(
          d_key, d_output_index, len, keys_, vals_, capacity_, counter_, empty_key, empty_val);

  // Launch dump kernel
  dump_kernel<KeyType, CounterType><<<(capacity_ - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0, stream>>>(
      d_unique_key, keys_, vals_, 0, capacity_, d_output_counter, empty_key);

  CK_CUDA_THROW_(cudaGetLastError());
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::clear(cudaStream_t stream) {
  // Device Restorer
  CudaDeviceContext dev_restorer;
  // Set to the device of this op
  CK_CUDA_THROW_(cudaSetDevice(dev_));

  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, stream>>>(
          keys_, vals_, counter_, capacity_, empty_key, empty_val, init_counter_val_);

  CK_CUDA_THROW_(cudaGetLastError());
}

template class unique_op<unsigned int, uint64_t, std::numeric_limits<unsigned int>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<long long, uint64_t, std::numeric_limits<long long>::max(),
                         std::numeric_limits<uint64_t>::max()>;
}  // namespace unique_op
}  // namespace HugeCTR
