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
#include <hps/database_backend_detail.hpp>
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

  HCTR_DISALLOW_COPY_AND_MOVE(RocksDBBackend);

  RocksDBBackend() = delete;

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

  size_t dump_bin(const std::string& table_name, std::ofstream& file) override;

  size_t dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

  size_t load_dump_sst(const std::string& table_name, const std::string& path) override;

 protected:
  inline rocksdb::ColumnFamilyHandle* get_column_handle_(const std::string& table_name) const {
    const auto& it{column_handles_.find(table_name)};
    return it != column_handles_.end() ? it->second : nullptr;
  }

  inline rocksdb::ColumnFamilyHandle* get_or_create_column_handle_(const std::string& table_name) {
    const auto& it{column_handles_.find(table_name)};
    if (it != column_handles_.end()) {
      return it->second;
    }

    rocksdb::ColumnFamilyHandle* ch;
    HCTR_ROCKSDB_CHECK(db_->CreateColumnFamily(column_family_options_, table_name, &ch));
    column_handles_.emplace(table_name, ch);
    return ch;
  }

  std::unique_ptr<rocksdb::DB> db_;
  std::unordered_map<std::string, rocksdb::ColumnFamilyHandle*> column_handles_;

  rocksdb::ColumnFamilyOptions column_family_options_;
  rocksdb::ReadOptions read_options_;
  rocksdb::WriteOptions write_options_;
  rocksdb::IngestExternalFileOptions ingest_file_options_;
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR