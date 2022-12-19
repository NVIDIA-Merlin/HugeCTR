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

#include <rocksdb/db.h>

#include <filesystem>
#include <hps/database_backend.hpp>
#include <unordered_map>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

struct RocksDBBackendParams final : public PersistentBackendParams {
  std::string path{"/tmp/rocksdb"};  // File-system path to the database.
  size_t num_threads{16};            // Number of threads that the RocksDB instance may use.
  bool read_only{
      false};  // If \p true will open the database in \p read-only mode. This allows simultaneously
               // querying the same RocksDB database from multiple clients.
};

/**
 * \p DatabaseBackend implementation that connects to a RocksDB to store/retrieve information (i.e.
 * harddisk storage).
 *
 * @tparam Key The data-type that is used for keys in this database.
 */
template <typename Key>
class RocksDBBackend final : public PersistentBackend<Key, RocksDBBackendParams> {
 public:
  using Base = PersistentBackend<Key, RocksDBBackendParams>;

  RocksDBBackend() = delete;
  DISALLOW_COPY_AND_MOVE(RocksDBBackend);

  /**
   * @brief Construct a new RocksDBBackend object.
   */
  RocksDBBackend(const RocksDBBackendParams& params);

  virtual ~RocksDBBackend();

  const char* get_name() const override { return "RocksDB"; }

  bool is_shared() const override { return false; }

  size_t size(const std::string& table_name) const override;

  size_t contains(const std::string& table_name, size_t num_keys, const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const Key* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const Key* keys,
               const DatabaseHitCallback& on_hit, const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const Key* keys, const DatabaseHitCallback& on_hit,
               const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const Key* keys) override;

  std::vector<std::string> find_tables(const std::string& model_name) override;

  void dump_bin(const std::string& table_name, std::ofstream& file) override;

  void dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

  void load_dump_sst(const std::string& table_name, const std::string& path) override;

 protected:
  rocksdb::DB* db_;
  std::unordered_map<std::string, rocksdb::ColumnFamilyHandle*> column_handles_;

  rocksdb::ColumnFamilyOptions column_family_options_;
  rocksdb::ReadOptions read_options_;
  rocksdb::WriteOptions write_options_;
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR