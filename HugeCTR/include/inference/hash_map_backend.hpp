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

#include <parallel_hashmap/phmap.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <inference/database_backend.hpp>
#include <shared_mutex>
#include <thread>
#include <thread_pool.hpp>
#include <unordered_map>
#include <vector>

// TODO: Significantly faster. Enable this by default?
// #define HCTR_EXPERIMENTAL_USE_BETTER_HASHMAP

namespace HugeCTR {

template <typename TPartition>
class HashMapBackendBase : public DatabaseBackend<typename TPartition::key_type> {
 public:
  using TBase = DatabaseBackend<typename TPartition::key_type>;
  using TKey = typename TPartition::key_type;

  HashMapBackendBase(size_t overflow_margin, DatabaseOverflowPolicy_t overflow_policy,
                     double overflow_resolution_target);

  virtual ~HashMapBackendBase() = default;

 protected:
  /**
   * Called internally in case a partition overflow is detected.
   */
  void resolve_overflow_(const std::string& table_name, size_t part_idx, TPartition& part,
                         size_t value_size) const;

  // Access control.
  mutable std::shared_mutex read_write_guard_;

  // Overflow-handling / pruning related parameters.
  const size_t overflow_margin_;
  const DatabaseOverflowPolicy_t overflow_policy_;
  const size_t overflow_resolution_target_;
};

/**
 * \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TPartition>
class HashMapBackend final : public HashMapBackendBase<TPartition> {
 public:
  using TBase = HashMapBackendBase<TPartition>;
  using TKey = typename TBase::TKey;

  /**
   * @brief Construct a new HashMapBackend object.
   * @param overflow_margin Margin at which further inserts will trigger overflow handling.
   * @param overflow_policy Policy to use in case an overflow has been detected.
   * @param overflow_resolution_target Target margin after applying overflow handling policy.
   */
  HashMapBackend(size_t overflow_margin = std::numeric_limits<size_t>::max(),
                 DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
                 double overflow_resolution_target = 0.8);

  const char* get_name() const override { return "HashMap"; }

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
  std::unordered_map<std::string, TPartition> tables_;
};

/**
 * Alternative \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory
 * that takes advantage of parallel processing capabilities.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TPartition>
class ParallelHashMapBackend final : public HashMapBackendBase<TPartition> {
 public:
  using TBase = HashMapBackendBase<TPartition>;
  using TKey = typename TBase::TKey;

  /**
   * @brief Construct a new parallelized HashMapBackend object.
   * @param num_partitions The number of parallel partitions.
   * @param overflow_margin Margin at which further inserts will trigger overflow handling.
   * @param overflow_policy Policy to use in case an overflow has been detected.
   * @param overflow_resolution_target Target margin after applying overflow handling policy.
   */
  ParallelHashMapBackend(
      size_t num_partitions, size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8);

  const char* get_name() const { return "ParallelHashMap"; }

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
  std::unordered_map<std::string, std::vector<TPartition>> tables_;
};

#define HCTR_DB_HASH_MAP_STL_(HMAP, DTYPE) HMAP<std::unordered_map<DTYPE, std::vector<char>>>
#define HCTR_DB_HASH_MAP_PHM_(HMAP, DTYPE) HMAP<phmap::flat_hash_map<DTYPE, std::vector<char>>>

}  // namespace HugeCTR