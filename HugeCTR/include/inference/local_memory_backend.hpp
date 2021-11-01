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
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <inference/database_backend.hpp>
#include <mutex>
#include <thread>
#include <thread_pool.hpp>
#include <unordered_map>
#include <vector>

// TODO: Significantly faster. Enable this by default?
// #define HCTR_EXPERIMENTAL_USE_BETTER_HASHMAP
#ifdef HCTR_EXPERIMENTAL_USE_BETTER_HASHMAP
#include <parallel_hashmap/phmap.h>
#endif

namespace HugeCTR {

/**
 * \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class LocalMemoryBackend final : public DatabaseBackend<TKey> {
 public:
  using TBase = DatabaseBackend<TKey>;
#ifdef HCTR_EXPERIMENTAL_USE_BETTER_HASHMAP
  using TMap = phmap::flat_hash_map<TKey, std::vector<char>>;
#else
  using TMap = std::unordered_map<TKey, std::vector<char>>;
#endif

  /**
   * @brief Construct a new LocalMemoryBackend object.
   */
  LocalMemoryBackend();

  const char* get_name() const override;

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const TKey* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const TKey* keys, char* values,
               size_t value_size, MissingKeyCallback& missing_callback) const override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const TKey* keys, char* values, size_t value_size,
               MissingKeyCallback& missing_callback) const override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const TKey* keys) override;

 protected:
  std::unordered_map<std::string, TMap> tables_;
};

/**
 * Alternative \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory
 * that takes advantage of parallel processing capabilities.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */

template <typename TKey>
class ParallelLocalMemoryBackend final : public DatabaseBackend<TKey> {
 public:
  using TBase = DatabaseBackend<TKey>;
#ifdef HCTR_EXPERIMENTAL_USE_BETTER_HASHMAP
  using TMap = phmap::flat_hash_map<TKey, std::vector<char>>;
#else
  using TMap = std::unordered_map<TKey, std::vector<char>>;
#endif

  /**
   * @brief Construct a new ParallelLocalMemoryBackend object.
   *
   * @param num_partitions Number of threads that can be utilized in parallel.
   */
  ParallelLocalMemoryBackend(size_t num_partitions);

  const char* get_name() const;

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const TKey* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const TKey* keys, char* values,
               size_t value_size, MissingKeyCallback& missing_callback) const override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const TKey* keys, char* values, size_t value_size,
               MissingKeyCallback& missing_callback) const override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const TKey* keys) override;

 protected:
  size_t num_partitions_;
  std::unordered_map<std::string, std::vector<TMap>> tables_;
};

}  // namespace HugeCTR