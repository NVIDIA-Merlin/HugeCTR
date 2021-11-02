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
#include <inference/rocksdb_backend.hpp>

namespace HugeCTR {

#define HCTR_ROCKSDB_CHECK(EXPR)                                                    \
  do {                                                                              \
    const rocksdb::Status _status = EXPR;                                           \
    HCTR_CHECK_HINT(_status.ok(), "RocksDB error: %s", _status.ToString().c_str()); \
  } while (0)

template <typename TKey>
RocksDBBackend<TKey>::RocksDBBackend(const std::string& path, const bool read_only,
                                     const size_t max_get_batch_size,
                                     const size_t max_set_batch_size)
    : TBase(),
      db_(nullptr),
      max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size) {
  HCTR_LOG(INFO, WORLD, "Connecting to RocksDB database...\n");

  rocksdb::Options options;
  options.create_if_missing = true;
  options.OptimizeLevelStyleCompaction();
  options.IncreaseParallelism();

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

    rocksdb::ColumnFamilyOptions column_options;
    column_options.OptimizeLevelStyleCompaction();
    for (const auto& column_name : column_names) {
      HCTR_LOG(INFO, WORLD, "RocksDB %s, found column family \"%s\".\n", path.c_str(),
               column_name.c_str());
      column_descriptors.emplace_back(column_name, column_options);
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
const char* RocksDBBackend<TKey>::get_name() const {
  return "RocksDB";
}

template <typename TKey>
size_t RocksDBBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                      const TKey* const keys) const {
  static const rocksdb::ReadOptions read_options;

  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::contains(table_name, num_keys, keys);
  }
  const auto& column_handle = column_handles_it->second;

  size_t hit_count = 0;

  switch (num_keys) {
    case 0: {
      break;
    }
    case 1: {
      const rocksdb::Slice k_slice(reinterpret_cast<const char*>(keys), sizeof(TKey));
      rocksdb::PinnableSlice v_slice;
      const auto& status = db_->Get(read_options, column_handle, k_slice, &v_slice);
      if (status.ok()) {
        hit_count++;
      } else {
        HCTR_CHECK_HINT(status.IsNotFound(), "RocksDB error: %s", status.ToString().c_str());
      }
      break;
    }
    default: {
      // Form query.
      const std::unordered_set<TKey> query(keys, &keys[num_keys]);

      // Iterate over column family and check entries 1 by 1.
      std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(read_options, column_handle));
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        HCTR_ROCKSDB_CHECK(it->status());

        const auto& k = it->key();
        HCTR_CHECK_HINT(k.size() == sizeof(TKey), "RocksDB return key size mismatch!");

        if (query.find(*reinterpret_cast<const TKey*>(k.data())) != query.end()) {
          hit_count++;
        }
      }
      break;
    }
  }

  HCTR_LOG(INFO, WORLD, "%s backend. Table: %s. Found %d / %d keys.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
bool RocksDBBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                  const TKey* keys, const char* values, const size_t value_size) {
  static const rocksdb::WriteOptions write_options;

  // Locate or create column family.
  rocksdb::ColumnFamilyHandle* column_handle;
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it != column_handles_.end()) {
    column_handle = column_handles_it->second;
  } else {
    rocksdb::ColumnFamilyOptions column_family_options;
    column_family_options.OptimizeLevelStyleCompaction();

    HCTR_ROCKSDB_CHECK(db_->CreateColumnFamily(column_family_options, table_name, &column_handle));

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

    HCTR_ROCKSDB_CHECK(db_->Write(write_options, &batch));
    num_inserts += batch.Count();
    HCTR_LOG(INFO, WORLD, "RocksDB table %s, query %d: Inserted %d pairs.\n", table_name.c_str(),
             num_queries, batch.Count());
  }

  HCTR_LOG(INFO, WORLD, "%s backend. Table: %s. Inserted %d / %d pairs.\n", get_name(),
           table_name.c_str(), num_inserts, num_pairs);
  return true;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                   const TKey* keys, char* values, const size_t value_size,
                                   MissingKeyCallback& missing_callback) const {
  static const rocksdb::ReadOptions read_options;

  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_keys, keys, values, value_size, missing_callback);
  }
  const auto& column_handle = column_handles_it->second;

  std::vector<rocksdb::ColumnFamilyHandle*> batch_handles;
  std::vector<rocksdb::Slice> batch_keys;
  std::vector<std::string> batch_values;

  size_t hit_count = 0;
  size_t num_queries = 0;

  const TKey* const keys_end = &keys[num_keys];
  for (; keys != keys_end; num_queries++) {
    // Create and launch query.
    batch_keys.clear();
    const TKey* const batch_end = std::min(&keys[max_get_batch_size_], keys_end);
    for (; keys != batch_end; keys++) {
      batch_keys.emplace_back(reinterpret_cast<const char*>(keys), sizeof(TKey));
    }
    batch_handles.resize(batch_keys.size(), column_handle);

    batch_values.clear();
    const auto& batch_status =
        db_->MultiGet(read_options, batch_handles, batch_keys, &batch_values);
    HCTR_CHECK(batch_status.size() == batch_values.size());

    // Process results.
    auto batch_values_it = batch_values.begin();
    for (const auto& status : batch_status) {
      if (status.ok()) {
        HCTR_CHECK_HINT(batch_values_it->size() == value_size,
                        "RocksDB table %s, return value embedding size mismatch! (%d != %d)",
                        table_name.c_str(), batch_values_it->size(), value_size);
        memcpy(values, batch_values_it->data(), value_size);
        hit_count++;
      } else if (status.IsNotFound()) {
        missing_callback(num_queries * max_get_batch_size_ +
                         (batch_values_it - batch_values.begin()));
      } else {
        HCTR_ROCKSDB_CHECK(status);
      }
      batch_values_it++;
      values += value_size;
    }

    HCTR_LOG(INFO, WORLD, "RocksDB table %s, query %d: Fetched %d keys. Hits %d.\n",
             table_name.c_str(), num_queries, batch_values.size(), hit_count);
  }

  HCTR_LOG(INFO, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
           table_name.c_str(), hit_count, num_keys);
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                   const size_t* indices, const TKey* const keys,
                                   char* const values, const size_t value_size,
                                   MissingKeyCallback& missing_callback) const {
  static const rocksdb::ReadOptions read_options;

  // Empty result, if database does not contain this column handle.
  const auto& column_handles_it = column_handles_.find(table_name);
  if (column_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, values, value_size,
                        missing_callback);
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
    const auto& batch_status =
        db_->MultiGet(read_options, batch_handles, batch_keys, &batch_values);
    HCTR_CHECK(batch_status.size() == batch_values.size());

    // Process results.
    auto batch_values_it = batch_values.begin();
    for (const auto& status : batch_status) {
      if (status.ok()) {
        HCTR_CHECK_HINT(batch_values_it->size() == value_size,
                        "RocksDB table %s, return value embedding size mismatch! (%d != %d)",
                        table_name.c_str(), batch_values_it->size(), value_size);
        memcpy(&values[*indices * value_size], batch_values_it->data(), value_size);
        hit_count++;
      } else if (status.IsNotFound()) {
        missing_callback(*indices);
      } else {
        HCTR_ROCKSDB_CHECK(status);
      }
      batch_values_it++;
      indices++;
    }
    HCTR_ASSERT(indices == batch_end);

    HCTR_LOG(INFO, WORLD, "RocksDB table %s, query %d: Fetched %d keys. Hits %d.\n",
             table_name.c_str(), num_queries, batch_values.size(), hit_count);
  }

  HCTR_LOG(INFO, WORLD, "%s backend. Table: %s. Fetched %d / %d values.\n", get_name(),
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

  HCTR_LOG(INFO, WORLD, "%s backend. Table %s erased (approximately %d pairs).\n", get_name(),
           table_name, hit_count);
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                   const TKey* keys) {
  static const rocksdb::WriteOptions write_options;

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

    HCTR_ROCKSDB_CHECK(db_->Write(write_options, &batch));
    HCTR_LOG(INFO, WORLD, "RocksDB table %s, query %d: Deleted %d keys. Hits %d.\n",
             table_name.c_str(), num_queries, batch.Count(), hit_count);
  }

  HCTR_LOG(INFO, WORLD, "%s backend. Table %s. %d / %d pairs erased.\n", get_name(),
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