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

#include <rocksdb/sst_file_reader.h>

#include <base/debug/logger.hpp>
#include <fstream>
#include <hps/database_backend.hpp>
#include <sstream>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename TKey>
DatabaseBackend<TKey>::DatabaseBackend(size_t max_get_batch_size, size_t max_set_batch_size)
    : max_get_batch_size_{max_get_batch_size}, max_set_batch_size_{max_set_batch_size} {}

template <typename TKey>
size_t DatabaseBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                       const TKey* keys,
                                       const std::chrono::nanoseconds& time_budget) const {
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                    const TKey* const keys, const DatabaseHitCallback& on_hit,
                                    const DatabaseMissCallback& on_miss,
                                    const std::chrono::nanoseconds& time_budget) {
  for (size_t i = 0; i < num_keys; i++) {
    on_miss(i);
  }
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                    const size_t* indices, const TKey* const keys,
                                    const DatabaseHitCallback& on_hit,
                                    const DatabaseMissCallback& on_miss,
                                    const std::chrono::nanoseconds& time_budget) {
  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; indices++) {
    on_miss(*indices);
  }
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::evict(const std::vector<std::string>& table_names) {
  size_t n = 0;
  for (const std::string& table_name : table_names) {
    n += evict(table_name);
  }
  return n;
}

template <typename TKey>
void DatabaseBackend<TKey>::dump(const std::string& table_name, const std::string& path,
                                 DBTableDumpFormat_t format) {
  // Resolve format if none specificed.
  if (format == DBTableDumpFormat_t::Automatic) {
    const std::string ext = std::filesystem::path(path).extension();
    if (ext == ".bin") {
      format = DBTableDumpFormat_t::Raw;
    } else if (ext == ".sst") {
      format = DBTableDumpFormat_t::SST;
    } else {
      HCTR_DIE("Unsupported file extension!");
    }
  }

  switch (format) {
    case DBTableDumpFormat_t::Raw: {
      std::ofstream file{path, std::ios::binary};

      // Write header.
      const char magic[] = "bin";
      file.write(magic, sizeof(magic));

      const uint32_t version = 1;
      file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

      const uint32_t key_size = sizeof(TKey);
      file.write(reinterpret_cast<const char*>(&key_size), sizeof(uint32_t));

      // Write data.
      dump_bin(table_name, file);
    } break;
    case DBTableDumpFormat_t::SST: {
      rocksdb::Options options;
      rocksdb::EnvOptions env_options;
      rocksdb::SstFileWriter file(env_options, options);
      HCTR_ROCKSDB_CHECK(file.Open(path));

      // Write data.
      dump_sst(table_name, file);

      HCTR_ROCKSDB_CHECK(file.Finish());
    } break;
    default: {
      HCTR_DIE("Unsupported DB table dump format!");
    } break;
  }
}

template <typename TKey>
void DatabaseBackend<TKey>::load_dump(const std::string& table_name, const std::string& path) {
  const std::string ext = std::filesystem::path(path).extension();
  if (ext == ".bin") {
    load_dump_bin(table_name, path);
  } else if (ext == ".sst") {
    load_dump_sst(table_name, path);
  } else {
    HCTR_DIE("Unsupported file extension!");
  }
}

template <typename TKey>
void DatabaseBackend<TKey>::load_dump_bin(const std::string& table_name, const std::string& path) {
  std::ifstream file{path, std::ios::binary};
  HCTR_CHECK(file.is_open());

  // Parse header.
  char magic[4];
  file.read(magic, sizeof(magic));
  HCTR_CHECK(file);
  HCTR_CHECK(magic[0] == 'b' && magic[1] == 'i' && magic[2] == 'n' && magic[3] == '\0');

  uint32_t version;
  file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
  HCTR_CHECK(file);
  HCTR_CHECK(version == 1);

  uint32_t key_size;
  file.read(reinterpret_cast<char*>(&key_size), sizeof(uint32_t));
  HCTR_CHECK(file);
  HCTR_CHECK(key_size == sizeof(TKey));

  uint32_t value_size;
  file.read(reinterpret_cast<char*>(&value_size), sizeof(uint32_t));
  if (file.eof()) {
    return;
  }
  HCTR_CHECK(file);

  std::vector<TKey> keys;
  keys.reserve(max_set_batch_size_);
  std::vector<char> values;
  values.reserve(max_set_batch_size_ * value_size);

  std::vector<char> tmp(std::max(sizeof(TKey), static_cast<size_t>(value_size)));
  while (!file.eof()) {
    // Read key.
    file.read(tmp.data(), sizeof(TKey));
    if (file.eof()) {
      break;
    }
    HCTR_CHECK(file.good());
    keys.emplace_back(*reinterpret_cast<TKey*>(tmp.data()));

    // Read value.
    file.read(tmp.data(), tmp.size());
    HCTR_CHECK(file.good() || file.eof());
    values.insert(values.end(), tmp.begin(), tmp.end());

    // Put batch into table.
    if (keys.size() >= max_set_batch_size_) {
      insert(table_name, keys.size(), keys.data(), values.data(), value_size);
      keys.clear();
      values.clear();
    }
  }

  // Fill remaining KVs into table.
  if (!keys.empty()) {
    insert(table_name, keys.size(), keys.data(), values.data(), value_size);
  }
}

template <typename TKey>
void DatabaseBackend<TKey>::load_dump_sst(const std::string& table_name, const std::string& path) {
  rocksdb::Options options;
  rocksdb::SstFileReader file{options};
  HCTR_ROCKSDB_CHECK(file.Open(path));

  rocksdb::ReadOptions read_options;
  std::unique_ptr<rocksdb::Iterator> it{file.NewIterator(read_options)};
  it->SeekToFirst();

  size_t value_size = 0;
  std::vector<TKey> keys;
  std::vector<char> values;

  for (; it->Valid(); it->Next()) {
    // Parse key.
    const rocksdb::Slice& k_view = it->key();
    HCTR_CHECK(k_view.size() == sizeof(TKey));
    const TKey k = *reinterpret_cast<const TKey*>(k_view.data());
    keys.emplace_back(k);

    // Parse value.
    const rocksdb::Slice& v_view = it->value();
    if (value_size == 0) {
      HCTR_CHECK(v_view.size() != 0);
      value_size = v_view.size();
    } else {
      HCTR_CHECK(v_view.size() == value_size);
    }
    const char* v = v_view.data();
    values.insert(values.end(), v, &v[value_size]);

    // If buffer full, insert.
    if (keys.size() >= max_set_batch_size_) {
      insert(table_name, keys.size(), keys.data(), values.data(), value_size);
      keys.clear();
      values.clear();
    }
  }

  // If buffer not yet empty.
  if (!keys.empty()) {
    insert(table_name, keys.size(), keys.data(), values.data(), value_size);
  }
}

template class DatabaseBackend<unsigned int>;
template class DatabaseBackend<long long>;

DatabaseBackendError::DatabaseBackendError(const std::string& backend, const size_t partition,
                                           const std::string& what)
    : backend_{backend}, partition_{partition}, what_{what} {}

std::string DatabaseBackendError::to_string() const {
  std::ostringstream os;
  os << backend_ << " DB Backend error (partition = " << partition_ << "): " << what_;
  return os.str();
}

template <typename TKey>
VolatileBackend<TKey>::VolatileBackend(const size_t max_get_batch_size,
                                       const size_t max_set_batch_size,
                                       const size_t overflow_margin,
                                       const DatabaseOverflowPolicy_t overflow_policy,
                                       const double overflow_resolution_target)
    : TBase(max_get_batch_size, max_set_batch_size),
      overflow_margin_(overflow_margin),
      overflow_policy_(overflow_policy),
      overflow_resolution_target_(hctr_safe_cast<size_t>(
          static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5)) {
  HCTR_CHECK(overflow_resolution_target_ <= overflow_margin_);
}

template <typename TKey>
std::future<void> VolatileBackend<TKey>::insert_async(
    const std::string& table_name, const std::shared_ptr<std::vector<TKey>>& keys,
    const std::shared_ptr<std::vector<char>>& values, size_t value_size) {
  HCTR_CHECK(keys->size() * value_size == values->size());
  return background_worker_.submit([this, table_name, keys, values, value_size]() {
    this->insert(table_name, keys->size(), keys->data(), values->data(), value_size);
  });
}

template <typename TKey>
void VolatileBackend<TKey>::synchronize() {
  background_worker_.await_idle();
}

template class VolatileBackend<unsigned int>;
template class VolatileBackend<long long>;

template <typename TKey>
PersistentBackend<TKey>::PersistentBackend(size_t max_get_batch_size, size_t max_set_batch_size)
    : TBase(max_get_batch_size, max_set_batch_size) {}

template class PersistentBackend<unsigned int>;
template class PersistentBackend<long long>;

}  // namespace HugeCTR