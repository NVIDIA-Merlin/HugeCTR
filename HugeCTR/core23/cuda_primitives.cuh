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

#pragma once

#include <cassert>
#include <core23/macros.hpp>
#include <core23/tensor_view.hpp>

namespace HugeCTR {
namespace core23 {

HCTR_DEVICE_INLINE int64_t atomic_cas(int64_t *address, int64_t compare, int64_t val) {
  return (int64_t)::atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

HCTR_DEVICE_INLINE uint64_t atomic_cas(uint64_t *address, uint64_t compare, uint64_t val) {
  return (uint64_t)::atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                               (unsigned long long)val);
}

HCTR_DEVICE_INLINE long long atomic_cas(long long *address, long long compare, long long val) {
  return (long long)atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

HCTR_DEVICE_INLINE unsigned int atomic_cas(unsigned int *address, unsigned int compare,
                                           unsigned int val) {
  return (unsigned int)atomicCAS((unsigned int *)address, (unsigned int)compare, (unsigned int)val);
}

HCTR_DEVICE_INLINE int atomic_cas(int *address, int compare, int val) {
  return (int)atomicCAS((int *)address, (int)compare, (int)val);
}

HCTR_DEVICE_INLINE long long atomic_add(long long *address, long long val) {
  return (long long)atomicAdd((unsigned long long *)address, (unsigned long long)val);
}

HCTR_DEVICE_INLINE int32_t atomic_add(int32_t *address, int32_t val) {
  return (int32_t)atomicAdd((unsigned int *)address, (unsigned int)val);
}

HCTR_DEVICE_INLINE uint32_t atomic_add(uint32_t *address, uint32_t val) {
  return (uint32_t)atomicAdd((unsigned int *)address, (unsigned int)val);
}

HCTR_DEVICE_INLINE int64_t atomic_add(int64_t *address, int64_t val) {
  return (int64_t)atomicAdd((unsigned long long int *)address, (unsigned long long int)val);
}

HCTR_DEVICE_INLINE uint64_t atomic_add(uint64_t *address, uint64_t val) {
  return (uint64_t)atomicAdd((unsigned long long int *)address, (unsigned long long int)val);
}

// TODO: existing CUDA primitives and util kernels must be moved to this header file including
// sinusoidal_kernel
// TODO: add VecT

template <typename Type>
__global__ void fill_kernel(Type *data, int64_t num_elements, const Type val) {
  const int64_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t num_threads = blockDim.x * gridDim.x;
  for (int64_t tid = tid_base; tid < num_elements; tid += num_threads) {
    data[tid] = val;
  }
}

template <typename DstType, typename SrcType, typename Op>
__global__ void transform_kernel(DstType *dst, const SrcType *src, int64_t num_elements, Op op) {
  const int64_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t num_threads = blockDim.x * gridDim.x;
  for (int64_t tid = tid_base; tid < num_elements; tid += num_threads) {
    dst[tid] = op(__ldg(src + tid));
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 1> input_tensor,
                            TensorView<BuiltInType, 1> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t x = x_base; x < output_tensor.size(0); x += blockDim.x * gridDim.x) {
    output_tensor[x] = input_tensor[x];
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 2> input_tensor,
                            TensorView<BuiltInType, 2> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t y_base = blockIdx.y * blockDim.y + threadIdx.y;
  for (int64_t y = y_base; y < output_tensor.size(0); y += blockDim.y * gridDim.y) {
    for (int64_t x = x_base; x < output_tensor.size(1); x += blockDim.x * gridDim.x) {
      output_tensor[y][x] = input_tensor[y][x];
    }
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 3> input_tensor,
                            TensorView<BuiltInType, 3> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t y_base = blockIdx.y * blockDim.y + threadIdx.y;
  for (int64_t z = 0; z < output_tensor.size(0); z++) {
    for (int64_t y = y_base; y < output_tensor.size(1); y += blockDim.y * gridDim.y) {
      for (int64_t x = x_base; x < output_tensor.size(2); x += blockDim.x * gridDim.x) {
        output_tensor[z][y][x] = input_tensor[z][y][x];
      }
    }
  }
}

template <typename KeyEntry, typename TableEntry, typename CounterType, typename Hash>
HCTR_DEVICE_INLINE CounterType get_insert_dump(const KeyEntry key, TableEntry *table,
                                               CounterType *d_global_counter, KeyEntry &unique_out,
                                               size_t capacity, KeyEntry empty_key,
                                               bool &is_unique) {
  using TableValue = typename TableEntry::value_type;
  using KeyType = typename TableEntry::key_type;
  CounterType current_idx = 0;
  size_t counter = 0;
  TableValue insert_value;
  size_t pos = Hash()(key) % capacity;
  uint32_t r_idx_plus_one = 0;
  while (true) {
    TableEntry current_slot = table[pos];
    KeyType *key_ptr = &table[pos].key;
    volatile uint64_t *table_value_ptr = &table[pos].value.value;
    // const KeyType old_key = atomicCAS(key_ptr, empty_key.key, key.store_idx());
    KeyType existing_key = table[pos].key;
    if (existing_key == empty_key.key) {
      const KeyType old_key = atomic_cas(key_ptr, empty_key.key, key.store_idx());
      if (old_key == empty_key.key) {
        current_idx = atomic_add(d_global_counter, 1);
        r_idx_plus_one = static_cast<uint32_t>(current_idx) + 1;
        insert_value.write(r_idx_plus_one, key);
        *table_value_ptr = insert_value.value;
        unique_out = key;
        is_unique = true;
        break;
      } else if (old_key == key.store_idx()) {
        insert_value.value = *table_value_ptr;
        while (insert_value.reverse_idx() == 0) {
          insert_value.value = *table_value_ptr;
        }
        if (key.match(insert_value)) {
          current_idx = insert_value.reverse_idx();
          r_idx_plus_one = static_cast<uint32_t>(current_idx);
          is_unique = false;
          break;
        } else {
          pos = (pos + 1) % capacity;
          counter = counter + 1;
        }
      }
    } else if (existing_key == key.store_idx()) {
      insert_value.value = *table_value_ptr;
      while (insert_value.reverse_idx() == 0) {
        insert_value.value = *table_value_ptr;
      }
      if (key.match(insert_value)) {
        current_idx = insert_value.reverse_idx();
        r_idx_plus_one = static_cast<uint32_t>(current_idx);
        is_unique = false;
        break;
      } else {
        pos = (pos + 1) % capacity;
        counter = counter + 1;
      }
    } else {
      pos = (pos + 1) % capacity;
      counter = counter + 1;
    }
    if (counter >= capacity) {
      assert(false && "error: unique op fails: hashtable is full");
      break;
    }
  }
  return r_idx_plus_one;
}

}  // namespace core23

}  // namespace HugeCTR
