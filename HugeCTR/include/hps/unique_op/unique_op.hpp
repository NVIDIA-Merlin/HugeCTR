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

// The unique op
template <typename KeyType, typename CounterType, KeyType empty_key, CounterType empty_val,
          typename hasher = MurmurHash3_32<KeyType>>
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

  // Counter for value index
  CounterType* counter_;
};

}  // namespace unique_op
}  // namespace HugeCTR
