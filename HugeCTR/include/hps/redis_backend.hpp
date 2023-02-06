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

#include <sw/redis++/redis++.h>

#include <hps/database_backend.hpp>
#include <memory>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

struct RedisClusterBackendParams final : public VolatileBackendParams {
  std::string address{"127.0.0.1:7000"};  // The host or IP address(es) of the Redis cluster.
                                          // Multiple addresses should be comma-separated.

  std::string user_name{"default"};  // Redis username.
  std::string password;              // Plaintext password of the user.

  size_t num_node_connections{5};  // Maximum number of simultaneous connections that are formed
                                   // with the same redis server node.

  bool enable_tls{
      false};  // If true, connections formed with server nodes will be secured using SSL/TLS.
  std::string ca_certificate{
      "cacertbundle.crt"};  // Path to a file or directory containing certificates of trusted CAs.
  std::string client_certificate{"client_cert.pem"};  // Certificate to use for this client.
  std::string client_key{"client_key.pem"};           // Private key to use for this client.
  std::string server_name_identification{
      "redis.localhost"};  // SNI to request (can deviate from connection address).
};

/**
 * \p DatabaseBackend implementation that connects to a Redis to store/retrieve information (i.e.
 * distributed storage).
 *
 * @tparam Key The data-type that is used for keys in this database.
 */
template <typename Key>
class RedisClusterBackend final : public VolatileBackend<Key, RedisClusterBackendParams> {
 public:
  using Base = VolatileBackend<Key, RedisClusterBackendParams>;

  HCTR_DISALLOW_COPY_AND_MOVE(RedisClusterBackend);

  RedisClusterBackend() = delete;

  /**
   * @brief Construct a new RedisClusterBackend object.
   */
  RedisClusterBackend(const RedisClusterBackendParams& params);

  virtual ~RedisClusterBackend();

  const char* get_name() const override { return "RedisCluster"; }

  bool is_shared() const override { return true; }

  size_t contains(const std::string& table_name, size_t num_keys, const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override;

  size_t size(const std::string& table_name) const override;

  size_t insert(const std::string& table_name, size_t num_pairs, const Key* keys,
                const char* values, uint32_t value_size, size_t value_stride) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const Key* keys, char* values,
               size_t value_stride, const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const Key* keys, char* values, size_t value_stride,
               const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const Key* keys) override;

  std::vector<std::string> find_tables(const std::string& model_name) override;

  std::vector<Key> keys(const std::string& table_name) const;

  uint32_t value_size_for(const std::string& table_name) const;

  size_t dump_bin(const std::string& table_name, std::ofstream& file) override;

  size_t dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

 protected:
  /**
   * Called internally during `insert` if insertion causes an overflow situation.
   */
  void resolve_overflow_(const std::string& table_name, size_t part_index, size_t part_size);

  /**
   * Called asynchronously by the `background_worker_` to refresh the metadata of certain entries.
   *
   * @param table_name Name of the affected table.
   * @param part_index Index of the part that is affected.
   * @param keys The keys which need a refresh.
   * @param amount Relative amount to add to LFU counters.
   */
  void refresh_metadata_lfu_inc_(const std::string& table_name, size_t part_index,
                                 const std::vector<Key>& keys, long long amount);
  /**
   * Called asynchronously by the `background_worker_` to refresh the metadata of certain entries.
   *
   * @param table_name Name of the affected table.
   * @param part_index Index of the part that is affected.
   * @param keys The keys which need a refresh.
   * @param amount Absolute amount to assign to LFU counters.
   */
  void refresh_metadata_lfu_set_(const std::string& table_name, size_t part_index,
                                 const std::vector<Key>& keys, long long amount);

  /**
   * Called asynchronously by the `background_worker_` to refresh the metadata of certain entries.
   *
   * @param table_name Name of the affected table.
   * @param part_index Index of the part that is affected.
   * @param keys The keys which need a refresh.
   * @param time Unix timestamp value to fill in.
   */
  void refresh_metadata_lru_(const std::string& table_name, size_t part_index,
                             const std::vector<Key>& keys, time_t time);

  void queue_metadata_refresh_(const std::string& table_name, size_t part_index,
                               std::shared_ptr<std::vector<Key>>&& keys);

 protected:
  std::unique_ptr<sw::redis::RedisCluster> redis_;

  // Worker used to update timestamps and carry out overflow handling.
  mutable ThreadPool background_worker_{"redis bg worker", 1};
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR