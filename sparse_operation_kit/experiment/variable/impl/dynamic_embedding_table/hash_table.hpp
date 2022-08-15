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

#pragma once
#include <concurrent_unordered_map.cuh>
#include <limits>

/**
 * The HashTable class is wrapped by cudf library for hash table operations on single GPU.
 * In this class, we implement the GPU version of the common used operations of hash table,
 * such as insert() / get() / set() / dump()...
 */
template <typename KeyType, typename ValType>
class HashTable {
  static constexpr KeyType empty_key = std::numeric_limits<KeyType>::max();

 public:
  /**
   * The constructor of HashTable.
   * @param capacity the number of <key,value> pairs in the hash table.
   * @param count the existed number of <key,value> pairs in the hash table.
   */
  HashTable(size_t capacity, size_t count = 0);

  /**
   * The destructor of HashTable.
   */
  ~HashTable();

  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  HashTable(const HashTable &) = delete;

  /**
   * The declaration for indicating that there is no default operator "=" overloading in this class.
   */
  HashTable &operator=(const HashTable &) = delete;

  /**
   * The insert function for hash table. "insert" means putting some new <key,value> pairs
   * into the current hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  void insert_or_assign(const KeyType *d_keys, const ValType *d_vals, size_t len,
                        cudaStream_t stream = 0);

  /**
   * The get function for hash table. "get" means fetching some values indexed
   * by the given keys.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got from the hash table.
   * @param stream the cuda stream for this operation.
   */
  void lookup(const KeyType *d_keys, ValType *d_vals, size_t len, cudaStream_t stream = 0) const;

  /**
   * The get_insert function for hash table. "get_insert" means if we can find
   * the keys in the hash table, the values indexed by the keys will be returned,
   * which is known as a "get" operation; Otherwise, the not-found keys together
   * with the values computed by the current device counter automatically will be
   * inserted into the hash table.
   * @param d_keys the device pointers for the keys.
   * @param d_vals the device pointers for the values.
   * @param len the number of <key,value> pairs to be got or inserted into the hash table.
   * @param stream the cuda stream for this operation.
   */
  void lookup_or_insert(const KeyType *d_keys, ValType *d_vals, size_t len,
                        cudaStream_t stream = 0);

  /**
   * Get the current size of the hash table. Size is also known as the number
   * of <key,value> pairs.
   * @param stream the cuda stream for this operation.
   */
  size_t get_size(cudaStream_t stream = 0) const;

  /**
   * Get the capacity of the hash table. "capacity" is known as the number of
   * <key,value> pairs of the hash table.
   */
  size_t get_capacity() const;

 private:
  static const int BLOCK_SIZE_ =
      256; /**< The block size of the CUDA kernels. The default value is 256. */

  const float LOAD_FACTOR = 0.75f;

  const size_t capacity_;

  concurrent_unordered_map<KeyType, ValType, std::numeric_limits<KeyType>::max()>
      *container_; /**< The object of the Table class which is defined in the
                      concurrent_unordered_map class. */

  // Counter for value index
  size_t *d_counter_; /**< The device counter for value index. */
  size_t *d_container_size_;
};
