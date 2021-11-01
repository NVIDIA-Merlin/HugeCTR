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

// Copy and modify part code from R3C,which is a C++ open source client for redis based on hiredis
// (https://github.com/redis/hiredis)
#include <hiredis/hiredis.h>

#include <inference/hierarchicaldb.hpp>
#include <inference/redis.hpp>
#include <inference/rocksdb.hpp>
#include <iostream>

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "

namespace HugeCTR {

template <typename TypeHashKey>
hierarchical_db<TypeHashKey>::hierarchical_db() : _connected(false) {}

template <typename TypeHashKey>
hierarchical_db<TypeHashKey>::hierarchical_db(redis<TypeHashKey>* redis_context,
                                              rocks_db<TypeHashKey>* rocksdb_context)
    : redis_context(redis_context), rocksdb_context(rocksdb_context), _connected(false) {}

template <typename TypeHashKey>
hierarchical_db<TypeHashKey>::hierarchical_db(parameter_server_config ps_config)
    : ps_config_(ps_config), _connected(false) {
  rocksdb_context = new rocks_db<TypeHashKey>(ps_config);
  redis_context = new redis<TypeHashKey>(ps_config);
}

template <typename TypeHashKey>
hierarchical_db<TypeHashKey>::~hierarchical_db() {
  if (_connected) {
    // DEBUG<<"free _context"<<std::endl;
    rocksdb_context->disconnect();
    redis_context->disconnect();
    _connected = false;
  }
}

template <typename TypeHashKey>
void hierarchical_db<TypeHashKey>::init(const std::string& KDBPath, const std::string& ip, int port,
                                        const std::string& pwd, const float cache_size_percentage) {
  _ip = ip;
  _port = port;
  _kdbpath = KDBPath;
  cache_size_percentage_ = cache_size_percentage;
  redis_context->init(ip, pwd);
  rocksdb_context->init(KDBPath);
  rc_client_t = new r3c::CRedisClient(ip, pwd);
}

template <typename TypeHashKey>
void hierarchical_db<TypeHashKey>::connect() {
  if (_connected) {
    // DEBUG
    return;
  }
  rocksdb_context->connect();
  // redis_context->connect();
  _connected = true;
}

template <typename TypeHashKey>
void hierarchical_db<TypeHashKey>::disconnect() {
  if (_connected) {
    // DEBUG<<"free _context"<<std::endl;
    rocksdb_context->disconnect();
    redis_context->disconnect();
    _connected = false;
  }
}

template <typename TypeHashKey>
void hierarchical_db<TypeHashKey>::load_data(const std::string& model_config_path) {
  return;
}

template <typename TypeHashKey>
void hierarchical_db<TypeHashKey>::look_up(const TypeHashKey* embeddingcolumns_ptr,
                                           size_t num_samples, float* h_embeddingoutputvector,
                                           const std::string& model_name,
                                           size_t embedding_table_id) {
  size_t model_id;
  auto model_id_iter = ps_config_.model_name_id_map_.find(model_name);
  if (model_id_iter != ps_config_.model_name_id_map_.end()) {
    model_id = model_id_iter->second;
  }
  size_t embedding_size = ps_config_.embedding_vec_size_[model_id][embedding_table_id];
  std::vector<float> result(num_samples * embedding_size, 0.0f);

  cmget(embeddingcolumns_ptr, result, num_samples, model_name, std::to_string(embedding_table_id),
        embedding_size);
  memcpy(h_embeddingoutputvector, result.data(), sizeof(float) * embedding_size * num_samples);
  return;
}

template <typename TypeHashKey>
bool hierarchical_db<TypeHashKey>::mset(const std::vector<TypeHashKey>& keys,
                                        std::vector<float>& values, const std::string modelname,
                                        const std::string tablename, size_t embedding_size) {
  if (redis_context->cmset(keys, values, modelname, tablename, embedding_size,
                           cache_size_percentage_) &&
      rocksdb_context->mset(keys, values, modelname, tablename, embedding_size)) {
    DEBUG << "Set key to Redis and RocksDB successfully" << std::endl;
    return true;
  }
  return false;
}

template <typename TypeHashKey>
bool hierarchical_db<TypeHashKey>::mget(const TypeHashKey* keys, std::vector<float>& values,
                                        size_t length, std::string modelname,
                                        const std::string tablename, size_t embedding_size) {
  if (_connected == false) {
    return false;
  }
  values.clear();
  size_t len = length;
  values.reserve(len);
  uint32_t argc = 2;
  size_t argvlen[len + 2];
  const char* argv[len + 2];

  char argv1[] = "HMGET";
  argv[0] = argv1;
  argvlen[0] = strlen("HMGET");

  const std::string table = modelname + tablename;
  std::cout << "Redis gets missing keys from model: " << modelname << " and table: " << tablename
            << std::endl;
  argv[1] = table.c_str();
  argvlen[1] = table.length();

  std::cout << "length:" << length << std::endl;
  for (size_t t = 0; t < len; t++) {
    argvlen[argc] = sizeof(TypeHashKey);
    argv[argc] = reinterpret_cast<const char*>(&keys[t]);
    argc++;
  }
  redisReply* _reply = (redisReply*)redis_context->execargv(argc, argv, argvlen);

  if (_reply && _reply->type == REDIS_REPLY_ARRAY) {
    values.clear();
    for (unsigned int i = 0; i < _reply->elements; i++) {
      std::string strTemp(_reply->element[i]->str, _reply->element[i]->len);
      if (strTemp.size() <= 0) {
        std::vector<float> default_emb_vec(embedding_size, 0);
        rocksdb_context->mget(&keys[i], default_emb_vec, 1, modelname, tablename, embedding_size);
        values.insert(values.end(), default_emb_vec.data(),
                      default_emb_vec.data() + embedding_size);
      } else {
        float* temp = (reinterpret_cast<float*>(_reply->element[i]->str));
        values.insert(values.end(), temp, temp + embedding_size);
      }
    }
  }
  return true;
}

template <typename TypeHashKey>
bool hierarchical_db<TypeHashKey>::cmget(const TypeHashKey* keys, std::vector<float>& values,
                                         size_t length, std::string modelname,
                                         const std::string tablename, size_t embedding_size) {
  if (length <= 0) {
    return true;
  }
  std::string pwd = "";
  r3c::CRedisClient rc_client(_ip, pwd);

  if (_connected == false) {
    return false;
  }
  const std::string table = modelname + tablename;
  size_t key_len = sizeof(TypeHashKey) + table.size();
  size_t prefix_len = table.size();
  char* buf = new char[key_len];
  std::cout << "Redis Cluster gets missing keys from model: " << modelname
            << " and table: " << tablename << std::endl;
  for (size_t t = 0; t < length; t++) {
    std::string key;
    memcpy(buf, table.data(), prefix_len);
    memcpy(buf + prefix_len, reinterpret_cast<const char*>(&keys[t]), sizeof(TypeHashKey));
    key.assign(buf, key_len);
    std::string result;
    if (rc_client_t->get(key, &result)) {
      const float* temp = reinterpret_cast<const float*>(result.c_str());
      memcpy(&values[t * embedding_size], temp, embedding_size * sizeof(float));
    } else {
      std::vector<float> default_emb_vec(embedding_size, 0);
      rocksdb_context->mget(&keys[t], default_emb_vec, 1, modelname, tablename, embedding_size);
      memcpy(&values[t * embedding_size], default_emb_vec.data(), embedding_size * sizeof(float));
    }
  }
  delete buf;
  return true;
}

template class hierarchical_db<unsigned int>;
template class hierarchical_db<long long>;
}  // namespace HugeCTR