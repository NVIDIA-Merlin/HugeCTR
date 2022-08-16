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

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename TKey>
RocksDBBackend<TKey>::RocksDBBackend(const std::string& path, const size_t num_threads,
                                     const bool read_only, const size_t max_get_batch_size,
                                     const size_t max_set_batch_size)
    : TBase(max_get_batch_size, max_set_batch_size), db_{nullptr} {
  HCTR_LOG(INFO, WORLD, "Connecting to RocksDB database...\n");

  // Basic behavior.
  rocksdb::Options options;
  options.create_if_missing = true;
  options.manual_wal_flush = true;
  options.OptimizeForPointLookup(8);
  options.OptimizeLevelStyleCompaction();
  HCTR_CHECK(num_threads <= std::numeric_limits<int>::max());
  options.IncreaseParallelism(static_cast<int>(num_threads));

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
    if (!status.ok()) {
      HCTR_LOG_S(ERROR, WORLD) << "RocksDB " << path << ": Listing column names failed!"
                               << std::endl;
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
      HCTR_LOG_S(INFO, WORLD) << "RocksDB " << path << ", found column family \"" << column_name
                              << "\"." << std::endl;
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
  // Empty result, if database does not contain this column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return -1;
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  // Query database.
  size_t approx_num_keys = -1;
  if (!db_->GetIntProperty(col_handle, rocksdb::DB::Properties::kEstimateNumKeys,
                           &approx_num_keys)) {
    HCTR_LOG_S(WARNING, WORLD) << "RocksDB key count estimation API reported error for table \""
                               << table_name << "\"!" << std::endl;
  }
  return approx_num_keys;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                      const TKey* keys,
                                      const std::chrono::nanoseconds& time_budget) const {
  const auto begin = std::chrono::high_resolution_clock::now();

  // Empty result, if database does not contain this column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return TBase::contains(table_name, num_keys, keys, time_budget);
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
        ign_count++;
        break;
      }

      // Create and launch query.
      const rocksdb::Slice k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
      rocksdb::PinnableSlice v_view;
      const auto& status = db_->Get(read_options_, col_handle, k_view, &v_view);

      // Process results.
      if (status.ok()) {
        hit_count++;
      } else if (!status.IsNotFound()) {
        HCTR_ROCKSDB_CHECK(status);
      }
    } break;
    default: {
      std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
      std::vector<rocksdb::Slice> k_views;
      std::vector<std::string> v_views;

      const TKey* const keys_end = &keys[num_keys];
      for (size_t num_queries = 0; keys != keys_end; num_queries++) {
        // Check time budget.
        const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
        if (elapsed >= time_budget) {
          HCTR_LOG_S(WARNING, WORLD)
              << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
          for (; keys != keys_end; keys++) {
            ign_count++;
          }
          break;
        }

        // Create and launch query.
        k_views.clear();
        const TKey* const batch_end = std::min(&keys[this->max_get_batch_size_], keys_end);
        k_views.reserve(batch_end - keys);

        for (; keys != batch_end; keys++) {
          k_views.emplace_back(reinterpret_cast<const char*>(keys), sizeof(TKey));
        }
        col_handles.resize(k_views.size(), col_handle);

        v_views.clear();
        v_views.reserve(col_handles.size());
        const std::vector<rocksdb::Status>& statuses =
            db_->MultiGet(read_options_, col_handles, k_views, &v_views);

        // Process results.
        size_t batch_hits = 0;
        for (const auto& status : statuses) {
          if (status.ok()) {
            batch_hits++;
          } else if (!status.IsNotFound()) {
            HCTR_ROCKSDB_CHECK(status);
          }
        }

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", query "
                                 << num_queries << ": " << batch_hits << " / " << statuses.size()
                                 << " hits. Time: " << elapsed.count() << " / "
                                 << time_budget.count() << " us." << std::endl;
        hit_count += batch_hits;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
bool RocksDBBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                  const TKey* keys, const char* values, const size_t value_size) {
  // Locate or create column family.
  rocksdb::ColumnFamilyHandle* col_handle;
  {
    const auto& handles_it = column_handles_.find(table_name);
    if (handles_it != column_handles_.end()) {
      col_handle = handles_it->second;
    } else {
      HCTR_ROCKSDB_CHECK(db_->CreateColumnFamily(column_family_options_, table_name, &col_handle));
      column_handles_.emplace(table_name, col_handle);
    }
  }

  size_t num_inserts = 0;

  switch (num_pairs) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      const rocksdb::Slice k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
      const rocksdb::Slice v_view{values, value_size};
      HCTR_ROCKSDB_CHECK(db_->Put(write_options_, col_handle, k_view, v_view));
    } break;
    default: {
      rocksdb::Slice k_view{nullptr, sizeof(TKey)};
      rocksdb::Slice v_view{nullptr, value_size};
      rocksdb::WriteBatch batch;

      const TKey* const keys_end = &keys[num_pairs];
      for (size_t num_queries = 0; keys != keys_end; num_queries++) {
        // Assemble batch.
        batch.Clear();
        const TKey* const batch_end = std::min(&keys[this->max_set_batch_size_], keys_end);
        for (; keys != batch_end; keys++, values += value_size) {
          k_view.data_ = reinterpret_cast<const char*>(keys);
          v_view.data_ = values;
          HCTR_ROCKSDB_CHECK(batch.Put(col_handle, k_view, v_view));
        }

        // Lodge batch.
        HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));
        num_inserts += batch.Count();

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", query "
                                 << num_queries << ": Inserted " << batch.Count() << " pairs."
                                 << std::endl;
      }
    } break;
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Inserted "
                           << num_inserts << " / " << num_pairs << " pairs." << std::endl;
  return true;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                   const TKey* keys, const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss,
                                   const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  // Empty result, if database does not contain this column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_keys, keys, on_hit, on_miss, time_budget);
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
        on_miss(0);
        ign_count++;
        break;
      }

      // Create and launch query.
      const rocksdb::Slice k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
      rocksdb::PinnableSlice v_view;
      const auto& status = db_->Get(read_options_, col_handle, k_view, &v_view);

      // Process results.
      if (status.ok()) {
        on_hit(0, v_view.data(), v_view.size());
        hit_count++;
      } else if (status.IsNotFound()) {
        on_miss(0);
      } else {
        HCTR_ROCKSDB_CHECK(status);
      }
    } break;
    default: {
      std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
      std::vector<rocksdb::Slice> k_views;
      std::vector<std::string> v_views;

      const TKey* const keys_end = &keys[num_keys];
      for (size_t num_batches = 0, idx = 0; keys != keys_end; num_batches++) {
        // Check time budget.
        const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
        if (elapsed >= time_budget) {
          HCTR_LOG_S(WARNING, WORLD)
              << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
          for (; idx < num_keys; idx++) {
            on_miss(idx);
            ign_count++;
          }
          break;
        }

        // Create and launch query.
        k_views.clear();
        const TKey* const batch_end = std::min(&keys[this->max_get_batch_size_], keys_end);
        k_views.reserve(batch_end - keys);

        for (; keys != batch_end; keys++) {
          k_views.emplace_back(reinterpret_cast<const char*>(keys), sizeof(TKey));
        }
        col_handles.resize(k_views.size(), col_handle);

        v_views.clear();
        v_views.reserve(col_handles.size());
        const std::vector<rocksdb::Status>& statuses =
            db_->MultiGet(read_options_, col_handles, k_views, &v_views);

        // Process results.
        size_t batch_hits = 0;
        auto v_it = v_views.begin();
        auto s_it = statuses.begin();
        for (; s_it != statuses.end(); idx++, v_it++, s_it++) {
          if (s_it->ok()) {
            on_hit(idx, v_it->data(), v_it->size());
            batch_hits++;
          } else if (s_it->IsNotFound()) {
            on_miss(idx);
          } else {
            HCTR_ROCKSDB_CHECK(*s_it);
          }
        }

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", query "
                                 << num_batches << ": " << batch_hits << " / " << statuses.size()
                                 << " hits. Time: " << elapsed.count() << " / "
                                 << time_budget.count() << " us." << std::endl;
        hit_count += batch_hits;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table: " << table_name << ": " << hit_count
                           << " / " << (num_keys - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                   const size_t* indices, const TKey* const keys,
                                   const DatabaseHitCallback& on_hit,
                                   const DatabaseMissCallback& on_miss,
                                   const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();

  // Empty result, if database does not contain this column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return TBase::fetch(table_name, num_indices, indices, keys, on_hit, on_miss, time_budget);
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  size_t hit_count = 0;
  size_t ign_count = 0;

  switch (num_indices) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      // Check time budget.
      const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
      if (elapsed >= time_budget) {
        HCTR_LOG_S(WARNING, WORLD)
            << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
        on_miss(*indices);
        ign_count++;
        break;
      }

      // Create and launch query.
      const rocksdb::Slice k_view{reinterpret_cast<const char*>(&keys[*indices]), sizeof(TKey)};
      rocksdb::PinnableSlice v_view;
      const auto& status = db_->Get(read_options_, col_handle, k_view, &v_view);

      // Process results.
      if (status.ok()) {
        on_hit(*indices, v_view.data(), v_view.size());
        hit_count++;
      } else if (status.IsNotFound()) {
        on_miss(*indices);
      } else {
        HCTR_ROCKSDB_CHECK(status);
      }
    } break;
    default: {
      std::vector<rocksdb::ColumnFamilyHandle*> col_handles;
      std::vector<rocksdb::Slice> k_views;
      std::vector<std::string> v_views;

      const size_t* const indices_end = &indices[num_indices];
      for (size_t num_batches = 0; indices != indices_end; num_batches++) {
        // Check time budget.
        const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
        if (elapsed >= time_budget) {
          HCTR_LOG_S(WARNING, WORLD)
              << get_name() << " backend; Table " << table_name << ": Timeout!" << std::endl;
          for (; indices != indices_end; indices++) {
            on_miss(*indices);
            ign_count++;
          }
          break;
        }

        // Create and launch query.
        k_views.clear();
        const size_t* const batch_end = std::min(&indices[this->max_get_batch_size_], indices_end);
        k_views.reserve(batch_end - indices);

        for (const size_t* i = indices; i != batch_end; i++) {
          k_views.emplace_back(reinterpret_cast<const char*>(&keys[*i]), sizeof(TKey));
        }
        col_handles.resize(k_views.size(), col_handle);

        v_views.clear();
        v_views.reserve(col_handles.size());
        const std::vector<rocksdb::Status>& statuses =
            db_->MultiGet(read_options_, col_handles, k_views, &v_views);

        // Process results.
        size_t batch_hits = 0;
        auto v_it = v_views.begin();
        auto s_it = statuses.begin();
        for (; indices != batch_end; v_it++, s_it++, indices++) {
          if (s_it->ok()) {
            on_hit(*indices, v_it->data(), v_it->size());
            batch_hits++;
          } else if (s_it->IsNotFound()) {
            on_miss(*indices);
          } else {
            HCTR_ROCKSDB_CHECK(*s_it);
          }
        }
        HCTR_CHECK(v_it == v_views.end() && s_it == statuses.end());

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", batch "
                                 << num_batches << ": " << batch_hits << " / " << statuses.size()
                                 << " hits. Time: " << elapsed.count() << " / "
                                 << time_budget.count() << " us." << std::endl;
        hit_count += batch_hits;
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << hit_count
                           << " / " << (num_indices - ign_count) << " hits, " << ign_count
                           << " ignored." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::evict(const std::string& table_name) {
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return 0;
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  // Estimate number of evicted keys.
  size_t approx_num_keys = -1;
  if (!db_->GetIntProperty(col_handle, rocksdb::DB::Properties::kEstimateNumKeys,
                           &approx_num_keys)) {
    HCTR_LOG_S(WARNING, WORLD) << "RocksDB key count estimation API reported error for table \""
                               << table_name << "\"!" << std::endl;
  }

  // Drop the entire table form the database.
  HCTR_ROCKSDB_CHECK(db_->DropColumnFamily(col_handle));
  HCTR_ROCKSDB_CHECK(db_->DestroyColumnFamilyHandle(col_handle));
  column_handles_.erase(table_name);

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name
                           << " erased (approximately " << approx_num_keys << " pairs)."
                           << std::endl;
  return approx_num_keys;
}

template <typename TKey>
size_t RocksDBBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                   const TKey* keys) {
  // Locate column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return 0;
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  switch (num_keys) {
    case 0: {
      // Do nothing ;-).
    } break;
    case 1: {
      const rocksdb::Slice k_view{reinterpret_cast<const char*>(keys), sizeof(TKey)};
      HCTR_ROCKSDB_CHECK(db_->Delete(write_options_, col_handle, k_view));
    }
    case 2: {
      rocksdb::Slice k_view{nullptr, sizeof(TKey)};
      rocksdb::WriteBatch batch;

      const TKey* const keys_end = &keys[num_keys];
      for (size_t num_batches = 0; keys != keys_end; num_batches++) {
        // Assemble batch.
        batch.Clear();
        const TKey* const batch_end = std::min(&keys[this->max_set_batch_size_], keys_end);
        for (; keys != batch_end; keys++) {
          k_view.data_ = reinterpret_cast<const char*>(keys);
          HCTR_ROCKSDB_CHECK(batch.Delete(col_handle, k_view));
        }

        // Lodge batch.
        HCTR_ROCKSDB_CHECK(db_->Write(write_options_, &batch));

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", batch "
                                 << num_batches << ": Deleted ? / " << batch.Count() << " pairs."
                                 << std::endl;
      }
    }
  }
  HCTR_ROCKSDB_CHECK(db_->FlushWAL(true));

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": Deleted ? / "
                           << num_keys << " pairs." << std::endl;
  return -1;
}

template <typename TKey>
std::vector<std::string> RocksDBBackend<TKey>::find_tables(const std::string& model_name) {
  const std::string& tag_prefix = HierParameterServerBase::make_tag_name(model_name, "", false);

  std::vector<std::string> matches;
  for (const auto& pair : column_handles_) {
    if (pair.first.find(tag_prefix) == 0) {
      matches.push_back(pair.first);
    }
  }
  return matches;
}

template <typename TKey>
void RocksDBBackend<TKey>::dump_bin(const std::string& table_name, std::ofstream& file) {
  // Locate the column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return;
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  // Get a pointer to the beginning of the table.
  std::unique_ptr<rocksdb::Iterator> it{db_->NewIterator(read_options_, col_handle)};
  it->SeekToFirst();

  // Write value size to file.
  const uint32_t value_size = it->Valid() ? static_cast<uint32_t>(it->value().size()) : 0;
  file.write(reinterpret_cast<const char*>(&value_size), sizeof(uint32_t));

  // Append the key/value pairs one by one.
  for (; it->Valid(); it->Next()) {
    // Key
    {
      const rocksdb::Slice& k_view = it->key();
      HCTR_CHECK(k_view.size() == sizeof(TKey));
      file.write(k_view.data(), sizeof(TKey));
    }
    // Value
    {
      const rocksdb::Slice& v_view = it->value();
      HCTR_CHECK(v_view.size() == value_size);
      file.write(v_view.data(), value_size);
    }
  }
}

template <typename TKey>
void RocksDBBackend<TKey>::dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) {
  // Locate the column handle.
  const auto& col_handles_it = column_handles_.find(table_name);
  if (col_handles_it == column_handles_.end()) {
    return;
  }
  rocksdb::ColumnFamilyHandle* const col_handle = col_handles_it->second;

  // Get a pointer to the beginning of the table.
  std::unique_ptr<rocksdb::Iterator> it{db_->NewIterator(read_options_, col_handle)};
  it->SeekToFirst();

  // Append the key/value pairs one by one.
  for (; it->Valid(); it->Next()) {
    HCTR_ROCKSDB_CHECK(file.Put(it->key(), it->value()));
  }
}

template <typename TKey>
void RocksDBBackend<TKey>::load_dump_sst(const std::string& table_name, const std::string& path) {
  // TBase::load_dump_sst(table_name, path);

  // Locate or create column family.
  rocksdb::ColumnFamilyHandle* col_handle;
  {
    const auto& handles_it = column_handles_.find(table_name);
    if (handles_it != column_handles_.end()) {
      col_handle = handles_it->second;
    } else {
      HCTR_ROCKSDB_CHECK(db_->CreateColumnFamily(column_family_options_, table_name, &col_handle));
      column_handles_.emplace(table_name, col_handle);
    }
  }

  // Call direction ingestion function.
  static rocksdb::IngestExternalFileOptions options;
  HCTR_ROCKSDB_CHECK(db_->IngestExternalFile(col_handle, {path}, options));
}

template class RocksDBBackend<unsigned int>;
template class RocksDBBackend<long long>;

}  // namespace HugeCTR