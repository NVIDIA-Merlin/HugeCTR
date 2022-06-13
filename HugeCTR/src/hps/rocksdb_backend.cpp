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

#include <base/debug/logger.hpp>
#include <hps/rocksdb_backend.hpp>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

#define HCTR_ROCKSDB_CHECK(EXPR)                                                    \
  do {                                                                              \
    const rocksdb::Status _status = (EXPR);                                         \
    HCTR_CHECK_HINT(_status.ok(), "RocksDB error: %s", _status.ToString().c_str()); \
  } while (0)

template <typename TKey>
RocksDBBackend<TKey>::RocksDBBackend(const std::string& path, const size_t num_threads,
                                     const bool read_only, const size_t max_get_batch_size,
                                     const size_t max_set_batch_size)
    : TBase(),
      db_(nullptr),
      max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size) {
  HCTR_LOG(INFO, WORLD, "Connecting to RocksDB database...\n");

  // Basic behavior.
  rocksdb::Options options;
  options.create_if_missing = true;
  options.manual_wal_flush = true;
  options.OptimizeForPointLookup(8);
  options.OptimizeLevelStyleCompaction();
  options.IncreaseParallelism(hctr_safe_cast<int>(num_threads));

  // Configure various behaviors and options used in later operations.
  column_family_options_.OptimizeForPointLookup(8);
  column_family_options_.OptimizeLevelStyleCompaction();
  // Need to tune: read_options_.readahead_size
  // Need to tune: read_options_.verify_checksums
  write_options_.sync = false;

  // Enumerate column families.
  std::vector<rocksdb::ColumnFamilyDescriptor> column_descriptors;
  {
    std::vector<std::string> column_names;
    const auto& status = rocksdb::DB::ListColumnFamilies(options, path, &column_names);
    if (!status.IsIOError()) {
      HCTR_ROCKSDB_CHECK(status);
    }
    bool has_default = false;
    for (const auto& column_name : column_names) {
      has_default |= column_name == rocksdb::kDefaultColumnFamilyName;
    }
    if (!has_default) {
      column_names.push_back(rocksdb::kDefaultColumnFamilyName);
    }

    for (const auto& column_name : column_names) {
      HCTR_LOG(INFO, WORLD, "RocksDB %s, found column family \"%s\".\n", path.c_str(),
               column_name.c_str());
      column_descriptors.emplace_back(column_name, column_family_options_);
    }
  }

  // Connect to DB with all column families.
  std::vector<rocksdb::ColumnFamilyHandle*> column_handles;
  if (read_only) {
    HCTR_ROCKSDB_CHECK(
        rocksdb::DB::OpenForReadOnly(options, path, column_descriptors, &column_handles, &db_));
  } else {
    HCTR_ROCKSDB_CHECK(rocksdb::DB::Open(options, path, column_descriptors, &column_handles, &db_));
  }
  HCTR_CHECK(column_handles.size() == column_descriptors.size());

  auto column_handles_it = column_handles.begin();
  for (const auto& column_descriptor : column_descriptors) {
    column_handles_.emplace(column_descriptor.name, *column_handles_it);
    column_handles_it++;
  }

  HCTR_LOG(INFO, WORLD, "Connected to RocksDB database!\n");
}

template <typename TKey>
RocksDBBackend<TKey>::~RocksDBBackend() {
  HCTR_LOG(INFO, WORLD, "Disconnecting from RocksDB database...\n");

  HCTR_ROCKSDB_CHECK(db_->SyncWAL());
  for (auto& ch : column_handles_) {
    HCTR_ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(ch.second));
  }
  column_handles_.clear();
  HCTR_ROCKSDB_CHECK(db_->Close());
  delete db_;
  db_ = nullptr;

  HCTR_LOG(INFO, WORLD, "Disconnected from RocksDB database!\n");
}

template <typename TKey>
size_t RocksDBBackend<TKey>::size(const std::string& table_name) const {
  size_t num_keys = -1;

  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it != column_handles_.end()) {
    if (!db_->GetIntProperty(rocksdb::DB::Properties::kEstimateNumKeys, &num_keys)) {
      HCTR_LOG_S(WARNING, WORLD) << "RocksDB key count estimation API reported error for table \""
                                 << table_name << "\"!" << std::endl;
    }
  }

  return num_keys;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                      const TKey* keys) const {
  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const auto& column_handle = column_handles_it->second;

  rocksdb::Slice k_slice(nullptr, sizeof(TKey));
  rocksdb::PinnableSlice v_slice;

  size_t hit_count = 0;

  // For now, just iterate over keys and try to find them.
  const TKey* const keys_end = &keys[num_keys];
  for (; keys != keys_end; keys++) {
    k_slice.data_ = reinterpret_cast<const char*>(keys);
    const auto& status = db_->Get(read_options_, column_handle, k_slice, &v_slice);
    if (status.ok()) {
      hit_count++;
    }
    if (!status.IsNotFound()) {
      HCTR_ROCKSDB_CHECK(status);
    }
  }

  HCTR_LOG(TRACE, WORLD, "%s backend. Table: %s. Found %d / %d keys.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
bool RocksDBBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                  const TKey* keys, const char* values, const size_t value_size) {
  // Locate or create column family.
  rocksdb::ColumnFamilyHandle* column_handle;
  const auto& handles_it = column_handles_.find(table_name);
  if (handles_it != column_handles_.end()) {
    column_handle = handles_it->second;
  } else {
    HCTR_ROCKSDB_CHECK(db_->CreateColumnFamily(column_family_options_, table_name, &column_handle));
    column_handles_.emplace(table_name, column_handle);
  }

  rocksdb::Slice key_slice(nullptr, sizeof(TKey));
  rocksdb::Slice value_slice(nullptr, value_size);
  rocksdb::WriteBatch batch;

  size_t num_queries = 0;
  size_t num_inserts = 0;

  const TKey* const keys_end = &keys[num_pairs];
  for (; keys != keys_end; num_queries++) {
    batch.Clear();
    const TKey* const batch_end = std::min(&keys[max_set_batch_size_], keys_end);
    for (; keys != batch_end; keys++, values += value_size) {
      key_slice.data_ = reinterpret_cast<const char*>(keys);
      value_slice.data_ = values;
      HCTR_ROCKSDB_CHECK(batch.Put(column_handle, key_slice, value_slice));
    }

    HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));
    num_inserts += batch.Count();
    HCTR_LOG(TRACE, WORLD, "RocksDB table %s, query %d: Inserted %d pairs.\n", table_name.c_str(),
             num_queries, batch.Count());
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG(TRACE, WORLD, "%s backend. Table: %s. Inserted %d / %d pairs.\n", get_name(),
           table_name.c_str(), num_inserts, num_pairs);
  return true;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                   const TKey* const keys, const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss) {
  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_keys, keys, on_hit, on_miss);
  }
  const auto& column_handle = column_handles_it->second;

  std::vector<rocksdb::ColumnFamilyHandle*> batch_handles;
  std::vector<rocksdb::Slice> batch_keys;
  std::vector<std::string> batch_values;

  size_t index = 0;
  size_t hit_count = 0;
  size_t num_queries = 0;

  const TKey* const keys_end = &keys[num_keys];
  for (const TKey* k = keys; k != keys_end; num_queries++) {
    // Create and launch query.
    batch_keys.clear();
    const TKey* const batch_end = std::min(&k[max_get_batch_size_], keys_end);
    for (; k != batch_end; k++) {
      batch_keys.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
    }
    batch_handles.resize(batch_keys.size(), column_handle);

    batch_values.clear();
    const std::vector<rocksdb::Status>& batch_status =
        db_->MultiGet(read_options_, batch_handles, batch_keys, &batch_values);

    // Process results.
    auto values_it = batch_values.begin();
    auto status_it = batch_status.begin();
    for (; status_it != batch_status.end(); status_it++, values_it++, index++) {
      if (status_it->ok()) {
        on_hit(index, values_it->data(), values_it->size());
        hit_count++;
      } else if (status_it->IsNotFound()) {
        on_miss(index);
      } else {
        HCTR_ROCKSDB_CHECK(*status_it);
      }
    }
    HCTR_CHECK(values_it == batch_values.end());

    HCTR_LOG(TRACE, WORLD, "%s backend; Table: %s, Query %d, Fetched %d keys, Hits %d.\n",
             get_name(), table_name.c_str(), num_queries, batch_values.size(), hit_count);
  }

  HCTR_LOG(TRACE, WORLD, "%s backend; Table: %s, Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                   const size_t* indices, const TKey* const keys,
                                   const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss) {
  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, on_hit, on_miss);
  }
  const auto& column_handle = column_handles_it->second;

  std::vector<rocksdb::ColumnFamilyHandle*> batch_handles;
  std::vector<rocksdb::Slice> batch_keys;
  std::vector<std::string> batch_values;

  size_t hit_count = 0;
  size_t num_queries = 0;

  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; num_queries++) {
    // Create and launch query.
    batch_keys.clear();
    const size_t* const batch_end = std::min(&indices[max_get_batch_size_], indices_end);
    for (const size_t* i = indices; i != batch_end; i++) {
      batch_keys.emplace_back(reinterpret_cast<const char*>(&keys[*i]), sizeof(TKey));
    }
    batch_handles.resize(batch_keys.size(), column_handle);

    batch_values.clear();
    const std::vector<rocksdb::Status>& batch_status =
        db_->MultiGet(read_options_, batch_handles, batch_keys, &batch_values);

    // Process results.
    auto values_it = batch_values.begin();
    auto status_it = batch_status.begin();
    for (; status_it != batch_status.end(); status_it++, values_it++, indices++) {
      if (status_it->ok()) {
        on_hit(*indices, values_it->data(), values_it->size());
        hit_count++;
      } else if (status_it->IsNotFound()) {
        on_miss(*indices);
      } else {
        HCTR_ROCKSDB_CHECK(*status_it);
      }
    }
    HCTR_CHECK(values_it == batch_values.end() && indices == batch_end);

    HCTR_LOG(TRACE, WORLD, "%s backend; Table: %s, Query %d, Fetched %d keys, Hits %d.\n",
             get_name(), table_name.c_str(), num_queries, batch_values.size(), hit_count);
  }

  HCTR_LOG(TRACE, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_indices);
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::evict(const std::string& table_name) {
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return 0;
  }

  size_t hit_count = -1;
  db_->GetIntProperty(rocksdb::DB::Properties::kEstimateNumKeys, &hit_count);

  HCTR_ROCKSDB_CHECK(db_->DropColumnFamily(column_handles_it->second));
  HCTR_ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(column_handles_it->second));
  column_handles_.erase(table_name);

  HCTR_LOG(TRACE, WORLD, "%s backend. Table %s erased (approximately %d pairs).\n", get_name(),
           table_name, hit_count);
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                   const TKey* keys) {
  // Locate column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return 0;
  }
  const auto& column_handle = column_handles_it->second;

  rocksdb::Slice key_slice(nullptr, sizeof(TKey));
  rocksdb::WriteBatch batch;

  const size_t hit_count = -1;
  size_t num_queries = 0;

  const TKey* const keys_end = &keys[num_keys];
  for (; keys != keys_end; num_queries++) {
    batch.Clear();
    const TKey* const batch_end = std::min(&keys[max_set_batch_size_], keys_end);
    for (; keys != batch_end; keys++) {
      key_slice.data_ = reinterpret_cast<const char*>(keys);
      HCTR_ROCKSDB_CHECK(batch.Delete(column_handle, key_slice));
    }

    HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));
    HCTR_LOG(TRACE, WORLD, "RocksDB table %s, query %d: Deleted %d keys. Hits %d.\n",
             table_name.c_str(), num_queries, batch.Count(), hit_count);
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG(TRACE, WORLD, "%s backend. Table %s. %d / %d pairs erased.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

#ifdef HCTR_ROCKSDB_CHECK
#undef HCTR_ROCKSDB_CHECK
#else
#error "HCTR_ROCKSDB_CHECK not defined?!"
#endif

template class RocksDBBackend<unsigned int>;
template class RocksDBBackend<long long>;

}  // namespace HugeCTR