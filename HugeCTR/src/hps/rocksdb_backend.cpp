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
#include <hps/hier_parameter_server_base.hpp>
#include <hps/rocksdb_backend.hpp>
#include <hps/rocksdb_backend_detail.hpp>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename Key>
RocksDBBackend<Key>::RocksDBBackend(const RocksDBBackendParams& params)
    : Base(params), db_{nullptr} {
  HCTR_LOG(INFO, WORLD, "Connecting to RocksDB database...\n");

  // Basic behavior.
  rocksdb::Options options;
  options.create_if_missing = true;
  options.manual_wal_flush = true;
  options.OptimizeForPointLookup(8);
  options.OptimizeLevelStyleCompaction();
  HCTR_CHECK(this->params_.num_threads <= std::numeric_limits<int>::max());
  options.IncreaseParallelism(static_cast<int>(this->params_.num_threads));

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
    const auto& status =
        rocksdb::DB::ListColumnFamilies(options, this->params_.path, &column_names);
    if (!status.ok()) {
      HCTR_LOG_C(ERROR, WORLD, "RocksDB ", this->params_.path, ": Listing column names failed!\n");
      column_names.clear();
    }
    bool has_default = false;
    for (const auto& column_name : column_names) {
      has_default |= column_name == rocksdb::kDefaultColumnFamilyName;
    }
    if (!has_default) {
      column_names.push_back(rocksdb::kDefaultColumnFamilyName);
    }

    for (const auto& column_name : column_names) {
      HCTR_LOG_C(INFO, WORLD, "RocksDB ", this->params_.path, ", found column family `",
                 column_name, "`.\n");
      column_descriptors.emplace_back(column_name, column_family_options_);
    }
  }

  // Connect to DB with all column families.
  std::vector<rocksdb::ColumnFamilyHandle*> column_handles;
  rocksdb::DB* db;
  if (this->params_.read_only) {
    HCTR_ROCKSDB_CHECK(rocksdb::DB::OpenForReadOnly(options, this->params_.path, column_descriptors,
                                                    &column_handles, &db));
  } else {
    HCTR_ROCKSDB_CHECK(
        rocksdb::DB::Open(options, this->params_.path, column_descriptors, &column_handles, &db));
  }
  db_.reset(db);
  HCTR_CHECK(column_handles.size() == column_descriptors.size());

  auto column_handles_it = column_handles.begin();
  for (const auto& column_descriptor : column_descriptors) {
    column_handles_.emplace(column_descriptor.name, *column_handles_it);
    column_handles_it++;
  }

  HCTR_LOG(INFO, WORLD, "Connected to RocksDB database!\n");
}

template <typename Key>
RocksDBBackend<Key>::~RocksDBBackend() {
  HCTR_LOG(INFO, WORLD, "Disconnecting from RocksDB database...\n");

  HCTR_ROCKSDB_CHECK(db_->SyncWAL());
  for (auto& ch : column_handles_) {
    HCTR_ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(ch.second));
  }
  column_handles_.clear();
  HCTR_ROCKSDB_CHECK(db_->Close());
  db_.reset();

  HCTR_LOG(INFO, WORLD, "Disconnected from RocksDB database!\n");
}

template <typename Key>
size_t RocksDBBackend<Key>::size(const std::string& table_name) const {
  rocksdb::ColumnFamilyHandle* const col_handle{get_column_handle_(table_name)};
  if (col_handle) {
    return 0;
  }

  // Query database.
  size_t approx_num_keys{0};
  if (!db_->GetIntProperty(col_handle, rocksdb::DB::Properties::kEstimateNumKeys,
                           &approx_num_keys)) {
    HCTR_LOG_C(WARNING, WORLD, "RocksDB key count estimation API reported error for table `",
               table_name, "`!");
  }
  return approx_num_keys;
}

template <typename Key>
size_t RocksDBBackend<Key>::contains(const std::string& table_name, const size_t num_keys,
                                     const Key* const keys,
                                     const std::chrono::nanoseconds& time_budget) const {
  const auto begin{std::chrono::high_resolution_clock::now()};

  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return Base::contains(table_name, num_keys, keys, time_budget);
  }

  size_t hit_count{0};
  size_t skip_count{0};

  std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
  std::vector<std::string> v_views;
  std::vector<rocksdb::Slice> k_views;
  k_views.reserve(std::min(num_keys, this->params_.max_batch_size));

  // Step through keys batch-by-batch.
  std::chrono::nanoseconds elapsed;
  const Key* const keys_end{&keys[num_keys]};
  for (const Key* k{keys}; k != keys_end;) {
    HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, nullptr);

    const size_t batch_size{std::min<size_t>(keys_end - k, this->params_.max_batch_size)};

    const size_t prev_hit_count{hit_count};
    if (![&]() {
          k_views.clear();
          HCTR_HPS_DB_APPLY_(SEQUENTIAL_DIRECT,
                             k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key)));
          col_handles.resize(k_views.size(), ch);

          v_views.clear();
          v_views.reserve(col_handles.size());
          const std::vector<rocksdb::Status>& statuses{
              db_->MultiGet(read_options_, col_handles, k_views, &v_views)};

          for (size_t idx{0}; idx < batch_size; ++idx) {
            const rocksdb::Status& s{statuses[idx]};
            if (s.ok()) {
              ++hit_count;
            } else if (!s.IsNotFound()) {
              HCTR_ROCKSDB_CHECK(s);
            }
          }

          return true;
        }()) {
      break;
    }

    HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ", batch ",
               (k - keys - 1) / this->params_.max_batch_size, ": ", hit_count - prev_hit_count,
               " / ", batch_size, " hits. Time: ", elapsed.count(), " / ", time_budget.count(),
               " ns.\n");
  }

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_keys - skip_count, " hits, ", skip_count, " skipped.\n");
  return hit_count;
}

template <typename Key>
size_t RocksDBBackend<Key>::insert(const std::string& table_name, const size_t num_pairs,
                                   const Key* const keys, const char* const values,
                                   const uint32_t value_size, const size_t value_stride) {
  HCTR_CHECK(value_size <= value_stride);

  rocksdb::ColumnFamilyHandle* const ch{get_or_create_column_handle_(table_name)};

  size_t num_inserts{0};

  rocksdb::WriteBatch batch;

  const Key* const keys_end = &keys[num_pairs];
  for (const Key* k{keys}; k != keys_end;) {
    const size_t batch_size{std::min<size_t>(keys_end - k, this->params_.max_batch_size)};

    if (![&]() {
          batch.Clear();
          HCTR_HPS_DB_APPLY_(
              SEQUENTIAL_DIRECT,
              HCTR_ROCKSDB_CHECK(batch.Put(ch, {reinterpret_cast<const char*>(k), sizeof(Key)},
                                           {&values[(k - keys) * value_stride], value_size})));
          HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));
          return true;
        }()) {
      break;
    }

    const size_t prev_num_inserts{num_inserts};
    num_inserts += batch.Count();

    HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ", batch ",
               (k - keys - 1) / this->params_.max_batch_size, ": Inserted ",
               num_inserts - prev_num_inserts, " + updated ",
               batch_size - num_inserts + prev_num_inserts, " = ", batch_size, " entries.\n");
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Inserted ", num_inserts,
             " + updated ", num_pairs - num_inserts, " = ", num_pairs, " entries.\n");
  return num_inserts;
}

template <typename Key>
size_t RocksDBBackend<Key>::fetch(const std::string& table_name, const size_t num_keys,
                                  const Key* const keys, char* const values,
                                  const size_t value_stride, const DatabaseMissCallback& on_miss,
                                  const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};

  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return Base::fetch(table_name, num_keys, keys, values, value_stride, on_miss, time_budget);
  }

  size_t miss_count{0};
  size_t skip_count{0};

  std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
  std::vector<std::string> v_views;
  std::vector<rocksdb::Slice> k_views;
  k_views.reserve(std::min(num_keys, this->params_.max_batch_size));

  // Step through input batch-by-batch.
  std::chrono::nanoseconds elapsed;
  const Key* const keys_end{&keys[num_keys]};
  for (const Key* k{keys}; k != keys_end;) {
    HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_DIRECT, on_miss);

    const size_t batch_size{std::min<size_t>(keys_end - k, this->params_.max_batch_size)};

    const size_t prev_miss_count{miss_count};
    if (!HCTR_HPS_ROCKSDB_FETCH_(SEQUENTIAL_DIRECT)) {
      break;
    }

    HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ", batch ",
               (k - keys - 1) / this->params_.max_batch_size, ": ",
               batch_size - miss_count + prev_miss_count, " / ", batch_size,
               " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
  }

  const size_t hit_count{num_keys - skip_count - miss_count};
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_keys - skip_count, " hits; skipped ", skip_count, " keys.\n");
  return hit_count;
}

template <typename Key>
size_t RocksDBBackend<Key>::fetch(const std::string& table_name, const size_t num_indices,
                                  const size_t* const indices, const Key* const keys,
                                  char* const values, const size_t value_stride,
                                  const DatabaseMissCallback& on_miss,
                                  const std::chrono::nanoseconds& time_budget) {
  const auto begin{std::chrono::high_resolution_clock::now()};

  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return Base::fetch(table_name, num_indices, indices, keys, values, value_stride, on_miss,
                       time_budget);
  }

  size_t miss_count{0};
  size_t skip_count{0};

  std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
  std::vector<std::string> v_views;
  std::vector<rocksdb::Slice> k_views;
  k_views.reserve(std::min(num_indices, this->params_.max_batch_size));

  std::chrono::nanoseconds elapsed;
  const size_t* const indices_end{&indices[num_indices]};
  for (const size_t* i{indices}; i != indices_end;) {
    HCTR_HPS_DB_CHECK_TIME_BUDGET_(SEQUENTIAL_INDIRECT, on_miss);

    const size_t batch_size{std::min<size_t>(indices_end - i, this->params_.max_batch_size)};

    const size_t prev_miss_count{miss_count};
    if (!HCTR_HPS_ROCKSDB_FETCH_(SEQUENTIAL_INDIRECT)) {
      break;
    }

    HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Partition ", table_name, ", batch ",
               (i - indices - 1) / this->params_.max_batch_size, ": ",
               v_views.size() - miss_count + prev_miss_count, " / ", v_views.size(),
               " hits. Time: ", elapsed.count(), " / ", time_budget.count(), " ns.\n");
  }

  const size_t hit_count{num_indices - skip_count - miss_count};
  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": ", hit_count, " / ",
             num_indices - skip_count, " hits; skipped ", skip_count, " keys.\n");
  return hit_count;
}

template <typename Key>
size_t RocksDBBackend<Key>::evict(const std::string& table_name) {
  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return 0;
  }

  // Estimate number of evicted keys.
  size_t num_entries{0};
  if (!db_->GetIntProperty(ch, rocksdb::DB::Properties::kEstimateNumKeys, &num_entries)) {
    HCTR_LOG_C(WARNING, WORLD, "RocksDB key count estimation API reported error for table `",
               table_name, "`!\n");
  }

  // Drop the entire table form the database.
  HCTR_ROCKSDB_CHECK(db_->DropColumnFamily(ch));
  HCTR_ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(ch));
  column_handles_.erase(table_name);

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ", num_entries,
             " entries (approximately).\n");
  return num_entries;
}

template <typename Key>
size_t RocksDBBackend<Key>::evict(const std::string& table_name, const size_t num_keys,
                                  const Key* const keys) {
  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return 0;
  }
  rocksdb::WriteBatch batch;

  const Key* const keys_end{&keys[num_keys]};
  for (const Key* k{keys}; k != keys_end;) {
    const size_t batch_size{std::min<size_t>(keys_end - k, this->params_.max_batch_size)};

    if (![&]() {
          batch.Clear();
          HCTR_HPS_DB_APPLY_(SEQUENTIAL_DIRECT,
                             HCTR_ROCKSDB_CHECK(batch.Delete(
                                 ch, {reinterpret_cast<const char*>(k), sizeof(Key)})));
          HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));
          return true;
        }()) {
      break;
    }

    HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ", batch ",
               (k - keys - 1) / this->params_.max_batch_size, ": Erased ? / ", batch_size,
               " entries.\n");
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG_C(TRACE, WORLD, get_name(), " backend; Table ", table_name, ": Erased ? / ", num_keys,
             " entries.\n");
  return 0;
}

template <typename Key>
std::vector<std::string> RocksDBBackend<Key>::find_tables(const std::string& model_name) {
  const std::string& tag_prefix{HierParameterServerBase::make_tag_name(model_name, "", false)};

  std::vector<std::string> table_names;
  for (const auto& pair : column_handles_) {
    if (pair.first.find(tag_prefix) == 0) {
      table_names.emplace_back(pair.first);
    }
  }
  return table_names;
}

template <typename Key>
size_t RocksDBBackend<Key>::dump_bin(const std::string& table_name, std::ofstream& file) {
  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return 0;
  }

  std::unique_ptr<rocksdb::Iterator> it{db_->NewIterator(read_options_, ch)};
  it->SeekToFirst();

  // Value size field.
  uint32_t value_size;
  if (it->Valid()) {
    value_size = static_cast<uint32_t>(it->value().size());
    HCTR_CHECK(value_size == it->value().size());
  } else {
    value_size = 0;
  }
  file.write(reinterpret_cast<const char*>(&value_size), sizeof(uint32_t));

  size_t num_entries{0};
  for (; it->Valid(); it->Next(), ++num_entries) {
    // Key
    {
      const rocksdb::Slice& k_view{it->key()};
      HCTR_CHECK(k_view.size() == sizeof(Key));
      file.write(k_view.data(), sizeof(Key));
    }
    // Value
    {
      const rocksdb::Slice& v_view{it->value()};
      HCTR_CHECK(v_view.size() == value_size);
      file.write(v_view.data(), value_size);
    }
  }
  return num_entries;
}

template <typename Key>
size_t RocksDBBackend<Key>::dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) {
  rocksdb::ColumnFamilyHandle* const ch{get_column_handle_(table_name)};
  if (!ch) {
    return 0;
  }

  std::unique_ptr<rocksdb::Iterator> it{db_->NewIterator(read_options_, ch)};
  it->SeekToFirst();

  size_t num_entries{0};
  for (; it->Valid(); it->Next(), ++num_entries) {
    HCTR_ROCKSDB_CHECK(file.Put(it->key(), it->value()));
  }
  return num_entries;
}

template <typename Key>
size_t RocksDBBackend<Key>::load_dump_sst(const std::string& table_name, const std::string& path) {
  // return Base::load_dump_sst(table_name, path);
  rocksdb::ColumnFamilyHandle* const ch{get_or_create_column_handle_(table_name)};
  HCTR_ROCKSDB_CHECK(db_->IngestExternalFile(ch, {path}, ingest_file_options_));
  return 0;
}

template class RocksDBBackend<unsigned int>;
template class RocksDBBackend<long long>;

}  // namespace HugeCTR