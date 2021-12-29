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

#include <sw/redis++/redis++.h>

#include <inference/database_backend.hpp>
#include <memory>
#include <thread_pool.hpp>

namespace HugeCTR {

/**
 * \p DatabaseBackend implementation that connects to a Redis to store/retrieve information (i.e.
 * distributed storage).
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class RedisClusterBackend final : public DatabaseBackend<TKey> {
 public:
  using TBase = DatabaseBackend<TKey>;

  /**
   * @brief Construct a new RedisClusterBackend object.
   *
   * @param address The host or IP address(es) of the Redis cluster. Multiple addresses should be
   * comma-separated.
   * @param password Password to submit upon successful connection.
   * @param num_partitions Number of Redis separate storage partitions. For achieving the best
   * performance, this should be signficantly higher than the number of cluster nodes! We use
   * modulo-N to assign partitions. Hence, you must not change this value after writing the first
   * data to a table.
   * @param max_get_batch_size Maximum number of key/value pairs that can participate in a reading
   * databse transaction.
   * @param max_set_batch_size Maximum number of key/value pairs that can participate in a writing
   * databse transaction.
   * @param overflow_margin Margin at which further inserts will trigger overflow handling.
   * @param overflow_policy Policy to use in case an overflow has been detected.
   * @param overflow_resolution_target Target margin after applying overflow handling policy.
   */
  RedisClusterBackend(
      const std::string& address, const std::string& user_name = "default",
      const std::string& password = "", size_t num_partitions = 8,
      size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000,
      size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8);

  virtual ~RedisClusterBackend();

  const char* get_name() const override { return "RedisCluster"; }

  bool is_shared() const override { return true; }

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const override;

  size_t max_capacity(const std::string& table_name) const override {
    return this->overflow_margin_ * num_partitions_;
  }

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
  const size_t
      num_partitions_;  // Do not change this value, after inserting data for the first time!
  mutable ThreadPool thread_pool_;
  const size_t max_get_batch_size_;
  const size_t max_set_batch_size_;
  std::unique_ptr<sw::redis::RedisCluster> redis_;

  // Pruning related options.
  const size_t overflow_margin_;
  const DatabaseOverflowPolicy_t overflow_policy_;
  const size_t overflow_resolution_target_;
};

}  // namespace HugeCTR