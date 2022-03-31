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

#include <hps/database_backend.hpp>
#include <memory>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * \p DatabaseBackend implementation that connects to a Redis to store/retrieve information (i.e.
 * distributed storage).
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class RedisClusterBackend final : public VolatileBackend<TKey> {
 public:
  using TBase = VolatileBackend<TKey>;

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
      bool refresh_time_after_fetch = false,
      size_t overflow_margin = std::numeric_limits<size_t>::max(),
      DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
      double overflow_resolution_target = 0.8);

  virtual ~RedisClusterBackend();

  const char* get_name() const override { return "RedisCluster"; }

  bool is_shared() const override { return true; }

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const override;

  size_t capacity(const std::string& table_name) const override {
    const size_t part_cap = this->overflow_margin_;
    const size_t total_cap = part_cap * num_partitions_;
    return (total_cap > part_cap) ? total_cap : part_cap;
  }

  size_t size(const std::string& table_name) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const TKey* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const TKey* keys,
               const DatabaseHitCallback& on_hit, const DatabaseMissCallback& on_miss) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const TKey* keys, const DatabaseHitCallback& on_hit,
               const DatabaseMissCallback& on_miss) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const TKey* keys) override;

 protected:
  /**
   * Called internally. Checks for overflow and initiate overflow handling in case a partition
   * overflow is detected.
   */
  void resolve_overflow_(const std::string& hkey_kv, const std::string& hkey_kt);

  /**
   * Called internally to reset a single timestamp.
   *
   * @param table_name Name of the table.
   * @param part Table partition number (not checked. Assumed to be correct!).
   * @param key The key for which to refresh the timestamp.
   */
  void touch_(const std::string& table_name, size_t part, const TKey& key, time_t time);

  /**
   * Called internally to reset many timestamps.
   *
   * @param table_name Name of the table.
   * @param part Table partition number (not checked. Assumed to be correct!).
   * @param keys The keys for which to refresh the timestamp.
   */
  void touch_(const std::string& table_name, size_t part,
              const std::shared_ptr<std::vector<TKey>>& keys, time_t time);

 protected:
  // Do not change this vector, after inserting data for the first time!
  const size_t num_partitions_;
  const size_t max_get_batch_size_;
  const size_t max_set_batch_size_;
  std::unique_ptr<sw::redis::RedisCluster> redis_;
  mutable ThreadPool workers_;
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR