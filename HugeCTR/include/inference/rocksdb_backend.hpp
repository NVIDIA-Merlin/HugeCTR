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

#include <inference/database_backend.hpp>
#include <unordered_map>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * \p DatabaseBackend implementation that connects to a RocksDB to store/retrieve information (i.e.
 * harddisk storage).
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class RocksDBBackend final : public PersistentBackend<TKey> {
 public:
  using TBase = PersistentBackend<TKey>;

  /**
   * @brief Construct a new RocksDBBackend object.
   *
   * @param path File-system path to the database.
   * @param num_threads Number of threads that the RocksDB may use.
   * @param read_only If \p true will open the database in \p read-only mode. This allows
   * simultaneously querying the same RocksDB database from multiple clients.
   * @param max_get_batch_size Maximum number of key/value pairs that can participate in a reading
   * databse transaction.
   * @param max_set_batch_size Maximum number of key/value pairs that can participate in a writing
   * databse transaction.
   */
  RocksDBBackend(const std::string& path, size_t num_threads = 16, bool read_only = false,
                 size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000);

  virtual ~RocksDBBackend();

  const char* get_name() const override { return "RocksDB"; }

  bool is_shared() const override { return false; }

  size_t size(const std::string& table_name) const override;

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const override;

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
  rocksdb::DB* db_;
  std::unordered_map<std::string, rocksdb::ColumnFamilyHandle*> column_handles_;
  const size_t max_get_batch_size_;
  const size_t max_set_batch_size_;

  rocksdb::ColumnFamilyOptions column_family_options_;
  rocksdb::ReadOptions read_options_;
  rocksdb::WriteOptions write_options_;
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR