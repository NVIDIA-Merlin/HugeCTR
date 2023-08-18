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

#include <hash_functions.cuh>
#include <utils.hpp>

#define UNIQUE_OP_LOAD_FACTOR 0.75

namespace HugeCTR {
namespace unique_op {

template <typename KeyType>
struct KeyEntry {
  KeyType key;
  HOST_DEVICE_INLINE KeyType store_idx() const { return (key); }
  HOST_DEVICE_INLINE KeyType flatten_idx() const { return (key); }
  template <typename TableValue>
  HOST_DEVICE_INLINE bool match(const TableValue insert_value) const {
    return ((uint32_t)key == insert_value.detail.key);
  }
};

union TableValue {
  uint64_t value;
  struct Detail {
    uint32_t r_idx;
    uint32_t key;
  } detail;
  template <typename KeyEntry>
  HOST_DEVICE_INLINE void write(uint32_t reverse_idx, KeyEntry key) {
    detail.r_idx = reverse_idx;
    detail.key = (uint32_t)key.flatten_idx();
  }
  HOST_DEVICE_INLINE uint32_t reverse_idx() const { return detail.r_idx; }
};

template <typename KeyType>
struct TableEntry {
  using key_type = KeyType;
  using value_type = TableValue;
  key_type key;
  value_type value;
};

struct Hash {
  HOST_DEVICE_INLINE size_t operator()(const KeyEntry<uint32_t>& key_entry) {
    using hash_func = MurmurHash3_32<uint32_t>;
    uint32_t key_hash = hash_func::hash(key_entry.key);
    return key_hash;
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyEntry<int32_t>& key_entry) {
    using hash_func = MurmurHash3_32<int32_t>;
    uint32_t key_hash = hash_func::hash(key_entry.key);
    return key_hash;
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyEntry<uint64_t>& key_entry) {
    using hash_func = MurmurHash3_32<uint64_t>;
    uint32_t key_hash = hash_func::hash(key_entry.key);
    return key_hash;
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyEntry<int64_t>& key_entry) {
    using hash_func = MurmurHash3_32<int64_t>;
    uint32_t key_hash = hash_func::hash(key_entry.key);
    return key_hash;
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyEntry<long long>& key_entry) {
    using hash_func = MurmurHash3_32<long long>;
    uint32_t key_hash = hash_func::hash(key_entry.key);
    return key_hash;
  }
};

// The unique op
template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher = Hash>
class unique_op {
 public:
  // Ctor
  unique_op(const size_t capacity, const CounterType init_counter_val = 0);

  // Dtor
  ~unique_op();

  // Get the max capacity of unique op obj
  size_t get_capacity() const;

  // Unique operation
  void unique(const KeyType* d_key, const size_t len, CounterType* d_output_index,
              KeyType* d_unique_key, size_t* d_output_counter, cudaStream_t stream);

  // Clear operation
  void clear(cudaStream_t stream);

 private:
  static const size_t BLOCK_SIZE_ = 64;

  // Capacity
  size_t capacity_;
  // CUDA device
  int dev_;
  // Init counter value
  CounterType init_counter_val_;

  // Keys and vals buffer
  KeyType* keys_;
  CounterType* vals_;
  TableEntry<KeyType>* table_;

  // Counter for value index
  CounterType* counter_;
};

}  // namespace unique_op
}  // namespace HugeCTR
