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

#include <inference/rocksdb.hpp>
#include <iostream>

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "

// Encapsulate the data to Slice
#define KEY_TO_SLICE(value, size) Slice(value, size)

#define VALUE_TO_SLICE(type, value, size) \
  rocksdb::Slice(reinterpret_cast<const char*>(&value), sizeof(type) * size)

// Get original data from Slice
#define FROM_STRING(type, value) *reinterpret_cast<const type*>(value.data())

namespace HugeCTR {

template <typename TypeHashKey>
rocks_db<TypeHashKey>::rocks_db() : _connected(false) {}

template <typename TypeHashKey>
rocks_db<TypeHashKey>::rocks_db(rocksdb::DB* db) : _db(db), _connected(false) {}

template <typename TypeHashKey>
rocks_db<TypeHashKey>::rocks_db(parameter_server_config ps_config)
    : ps_config_(ps_config), _connected(false) {}

template <typename TypeHashKey>
rocks_db<TypeHashKey>::~rocks_db() {
  if (_connected) {
    // DEBUG<<"free _context"<<std::endl;
    delete _db;
    _connected = false;
  }
}

template <typename TypeHashKey>
void rocks_db<TypeHashKey>::init(const std::string& KDBPath) {
  _kdbpath = KDBPath;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  _options.IncreaseParallelism();
  _options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  _options.create_if_missing = true;
}

template <typename TypeHashKey>
bool rocks_db<TypeHashKey>::isError(rocksdb::Status s) {
  if (s.ok()) {
    return true;
  }
  return false;
}

template <typename TypeHashKey>
void rocks_db<TypeHashKey>::connect() {
  if (_connected) {
    // DEBUG
    return;
  }
  rocksdb::Status s = rocksdb::DB::Open(_options, _kdbpath, &_db);
  if (!s.ok()) {
    std::cout << "rocks open fail!" << std::endl;
    exit(EXIT_FAILURE);
  }
  _connected = true;
}

template <typename TypeHashKey>
void rocks_db<TypeHashKey>::disconnect() {
  if (_connected) {
    // DEBUG<<"free _context"<<std::endl;
    delete _db;
    _connected = false;
  }
}

template <typename TypeHashKey>
void rocks_db<TypeHashKey>::load_data(const std::string& model_config_path) {
  return;
}

template <typename TypeHashKey>
void rocks_db<TypeHashKey>::look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
                                    float* h_embeddingoutputvector, const std::string& model_name,
                                    size_t embedding_table_id) {
  size_t model_id;
  auto model_id_iter = ps_config_.model_name_id_map_.find(model_name);
  if (model_id_iter != ps_config_.model_name_id_map_.end()) {
    model_id = model_id_iter->second;
  }
  size_t embedding_size = ps_config_.embedding_vec_size_[model_id][embedding_table_id];
  std::vector<float> result(num_samples * embedding_size, 0.0f);

  mget(embeddingcolumns_ptr, result, num_samples, model_name, std::to_string(embedding_table_id),
       embedding_size);
  memcpy(h_embeddingoutputvector, result.data(), sizeof(float) * embedding_size * num_samples);
  return;
}

template <typename TypeHashKey>
bool rocks_db<TypeHashKey>::mset(const std::vector<TypeHashKey>& keys, std::vector<float>& values,
                                 const std::string modelname, const std::string tablename,
                                 size_t embedding_size) {
  size_t num_key = 10000;
  size_t iter = keys.size() / num_key;
  if (_db == NULL) {
    return false;
  }

  if (keys.size() != values.size() / embedding_size) {
    return false;
  }

  const std::string table = modelname + tablename;
  std::cout << "Local RocksDB is initializing the embedding table: " << table << std::endl;
  size_t key_len = sizeof(TypeHashKey) + table.size();
  size_t prefix_len = table.size();

  for (size_t j = 0; j < iter; j++) {
    rocksdb::WriteBatch batch;
    char* buf = new char[key_len];
    std::vector<char*> bufs(num_key, buf);
    size_t index = j * num_key;
    bufs.clear();
    for (size_t t = 0; t < num_key; t++) {
      memcpy(bufs[t], table.data(), prefix_len);
      memcpy(bufs[t] + prefix_len, reinterpret_cast<const char*>(&keys[t + index]),
             sizeof(TypeHashKey));
      batch.Put(rocksdb::Slice(bufs[t], key_len),
                VALUE_TO_SLICE(float, values[(index + t) * embedding_size], embedding_size));
    }
    rocksdb::Status s = _db->Write(rocksdb::WriteOptions(), &batch);
    if (s.ok()) {
      bufs.clear();
      batch.Clear();
    } else {
      std::cout << "Iteration " << j << " insert fail, Please check RocksDB Status!" << std::endl;
    }
  }
  if ((iter + 1) * num_key >= keys.size()) {
    size_t rest_keys = keys.size() - iter * num_key;
    size_t index = iter * num_key;
    rocksdb::WriteBatch batch;
    char* buf = new char[key_len];
    std::vector<char*> bufs(rest_keys, buf);

    for (size_t t = 0; t < rest_keys; t++) {
      memcpy(bufs[t], table.data(), prefix_len);
      memcpy(bufs[t] + prefix_len, reinterpret_cast<const char*>(&keys[t + index]),
             sizeof(TypeHashKey));
      batch.Put(rocksdb::Slice(bufs[t], key_len),
                VALUE_TO_SLICE(float, values[(index + t) * embedding_size], embedding_size));
    }
    rocksdb::Status s = _db->Write(rocksdb::WriteOptions(), &batch);
    if (s.ok()) {
      bufs.clear();
      std::cout << "Last Iteration insert successfully" << std::endl;
    }
  }

  return true;
}

template <typename TypeHashKey>
bool rocks_db<TypeHashKey>::mget(const TypeHashKey* keys, std::vector<float>& values, size_t length,
                                 std::string modelname, const std::string tablename,
                                 size_t embedding_size) {
  if (length <= 0) {
    return true;
  }
  if (_db == NULL) {
    return false;
  }
  values.clear();
  size_t len = length;
  values.reserve(len);
  const std::string table = modelname + tablename;
  std::cout << "Rocksdb gets missing keys from model: " << modelname << " and table: " << tablename
            << std::endl;
  size_t key_len = sizeof(TypeHashKey) + table.size();
  size_t prefix_len = table.size();

  char* buf = new char[key_len];
  std::vector<char*> bufs(len, buf);
  std::vector<rocksdb::Slice> slices;

  bufs.clear();
  for (size_t t = 0; t < len; t++) {
    std::string results;
    memcpy(bufs[t], table.data(), prefix_len);
    memcpy(bufs[t] + prefix_len, reinterpret_cast<const char*>(&keys[t]), sizeof(TypeHashKey));
    slices.emplace_back(rocksdb::Slice(bufs[t]));
    rocksdb::Status s =
        _db->Get(rocksdb::ReadOptions(), rocksdb::Slice(bufs[t], key_len), &results);
    if (!s.ok()) {
      std::vector<float> default_emb_vec(embedding_size, 0);
      values.insert(values.end(), default_emb_vec.data(), default_emb_vec.data() + embedding_size);
    } else {
      const float* temp = (reinterpret_cast<const float*>(results.data()));
      values.insert(values.end(), temp, temp + embedding_size);
    }
  }
  return true;
}

template class rocks_db<unsigned int>;
template class rocks_db<long long>;
}  // namespace HugeCTR