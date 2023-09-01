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

#include <embedding/operators/unique_op.hpp>

namespace embedding {

__forceinline__ __device__ long atomicAdd(long* address, long val) {
  return (long)::atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long* address, long long val) {
  return (long long)::atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return (unsigned long)::atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ uint32_t atomicCAS(uint32_t* address, uint32_t compare, uint32_t val) {
  return (uint32_t)::atomicCAS((unsigned int*)address, (unsigned int)compare, (unsigned int)val);
}

__forceinline__ __device__ int32_t atomicCAS(int32_t* address, int32_t compare, int32_t val) {
  return (int32_t)::atomicCAS((int*)address, (int)compare, (int)val);
}

__forceinline__ __device__ uint64_t atomicCAS(uint64_t* address, uint64_t compare, uint64_t val) {
  return (uint64_t)::atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                               (unsigned long long)val);
}

__forceinline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)::atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

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
                            const size_t offset, const size_t search_length,
                            uint64_t* d_dump_counter, const KeyType empty_key) {
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
    atomicAdd(d_dump_counter, (uint64_t)block_acc);
  }

  // Each thread store one slot's data back to global memory, d_dump_counter is how many slots in
  // total dumped.
  if (valid_slot) {
    d_key[read_val] = read_key;
  }
}

template <typename KeyType, typename CounterType, typename hasher>
__global__ void get_insert_kernel(const KeyType* d_key, KeyType* d_unique_key, CounterType* d_val,
                                  const size_t len, KeyType* keys, CounterType* vals,
                                  const size_t capacity, CounterType* d_global_counter,
                                  const KeyType empty_key, const CounterType empty_val) {
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
      // const KeyType old_key = atomicCAS(keys + hash_index, empty_key, target_key);
      const KeyType existing_key = keys[hash_index];
      volatile CounterType& target_val_pos = vals[hash_index];
      if (empty_key == existing_key) {
        const KeyType old_key = atomicCAS(keys + hash_index, empty_key, target_key);
        if (empty_key == old_key) {
          CounterType result_val;
          result_val = atomicAdd(d_global_counter, 1);
          d_unique_key[result_val] = target_key;
          d_val[idx] = result_val;
          target_val_pos = result_val;
          break;
        } else if (target_key == old_key) {
          while (target_val_pos == empty_val) {
          };
          d_val[idx] = target_val_pos;
          break;
        }
      } else if (target_key == existing_key) {
        while (target_val_pos == empty_val) {
        };
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
    std::shared_ptr<CoreResourceManager> core, const size_t capacity,
    const CounterType init_counter_val)
    : core_(core), capacity_(capacity), init_counter_val_(init_counter_val) {
  // Check parameter
  if (capacity_ == 0) {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Invalid value for unique_op capacity");
    return;
  }

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  keys_ = core23::Tensor(params.shape({static_cast<int64_t>(capacity_)})
                             .data_type(core23::ToScalarType<KeyType>::value));
  vals_ = core23::Tensor(params.shape({static_cast<int64_t>(capacity_)})
                             .data_type(core23::ToScalarType<CounterType>::value));

  counter_ = core23::Tensor(
      params.shape({static_cast<int64_t>(1)}).data_type(core23::ToScalarType<CounterType>::value));

  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType><<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
      keys_.data<KeyType>(), vals_.data<CounterType>(), counter_.data<CounterType>(), capacity_,
      empty_key, empty_val, init_counter_val_);

  // Wait for initialization to finish
  HCTR_LIB_THROW(cudaStreamSynchronize(0));
  HCTR_LIB_THROW(cudaGetLastError());
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
  // Set the d_output_counter to 0
  HCTR_LIB_THROW(cudaMemsetAsync(d_output_counter, 0, sizeof(size_t), stream));

  if (len == 0) {
    return;
  }

  // Launch get_insert kernel to do unique
  get_insert_kernel<KeyType, CounterType, hasher>
      <<<(len - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0, stream>>>(
          d_key, d_unique_key, d_output_index, len, keys_.data<KeyType>(),
          vals_.data<CounterType>(), capacity_, counter_.data<CounterType>(), empty_key, empty_val);
  cudaMemcpyAsync(d_output_counter, counter_.data<CounterType>(), sizeof(CounterType),
                  cudaMemcpyDeviceToDevice, stream);
  // Launch dump kernel
  // dump_kernel<KeyType, CounterType><<<(capacity_ - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0,
  // stream>>>(
  //    d_unique_key, keys_.data<KeyType>(), vals_.data<CounterType>(), 0, capacity_,
  //    d_output_counter, empty_key);

  HCTR_LIB_THROW(cudaGetLastError());
}

template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher>
void unique_op<KeyType, CounterType, empty_key, empty_val, hasher>::clear(cudaStream_t stream) {
  // Initialization kernel, set all entry to unused <K,V>, set counter to init value
  init_kernel<KeyType, CounterType>
      <<<((capacity_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, stream>>>(
          keys_.data<KeyType>(), vals_.data<CounterType>(), counter_.data<CounterType>(), capacity_,
          empty_key, empty_val, init_counter_val_);

  HCTR_LIB_THROW(cudaGetLastError());
}

template class unique_op<int64_t, uint64_t, std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<uint64_t, uint64_t, std::numeric_limits<uint64_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<uint32_t, uint64_t, std::numeric_limits<uint32_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
template class unique_op<int32_t, uint64_t, std::numeric_limits<int32_t>::max(),
                         std::numeric_limits<uint64_t>::max()>;
}  // namespace embedding
