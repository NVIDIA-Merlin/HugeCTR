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

#include <unistd.h>

#include <inference/redis.hpp>
#include <iostream>

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "
#define REDIS_REPLY_STRING 1
#define REDIS_REPLY_ARRAY 2
#define REDIS_REPLY_INTEGER 3
#define REDIS_REPLY_NIL 4
#define REDIS_REPLY_STATUS 5
#define REDIS_REPLY_ERROR 6

namespace HugeCTR {

template <typename TypeHashKey>
redis<TypeHashKey>::redis() : _context(NULL), _reply(NULL), _connected(false) {}

template <typename TypeHashKey>
redis<TypeHashKey>::redis(redisContext* context) : _context(context), _connected(false) {}

template <typename TypeHashKey>
redis<TypeHashKey>::redis(parameter_server_config ps_config)
    : ps_config_(ps_config), _context(NULL), _reply(NULL), _connected(false) {}

template <typename TypeHashKey>
redis<TypeHashKey>::~redis() {
  if (_connected) {
    DEBUG << "free _context" << std::endl;
    ::redisFree(_context);
    _connected = false;
  }
}

template <typename TypeHashKey>
void redis<TypeHashKey>::init(const std::string& ip, const std::string& pwd) {
  std::string::size_type comma_pos = ip.find(',');
  if (comma_pos == std::string::npos) {
    const std::string::size_type colon_pos = ip.find(':');
    _ip = ip.substr(0, colon_pos);
    _port = (int32_t)std::stoi(ip.substr(colon_pos + 1).c_str());
  } else {
    _ip = ip;
  }
  _pwd = pwd;
}

template <typename TypeHashKey>
bool redis<TypeHashKey>::isError(redisReply* reply) {
  if (NULL == reply) {
    return true;
  }
  return false;
}

template <typename TypeHashKey>
bool redis<TypeHashKey>::exec(const std::string& cmd) {
  if (_context == NULL) {
    return false;
  }
  redisReply* _reply = (redisReply*)::redisCommand(_context, cmd.c_str());
  if (isError(_reply)) {
    freeReply(_reply);
    return false;
  }
  return true;
}

template <typename TypeHashKey>
redisReply* redis<TypeHashKey>::execargv(int argc, const char** argv, const size_t* argvlen) {
  return (redisReply*)redisCommandArgv(_context, argc, argv, argvlen);
}

template <typename TypeHashKey>
bool redis<TypeHashKey>::exec(const std::string& cmd, std::vector<std::string>& values) {
  values.clear();
  if (_context == NULL) {
    return false;
  }
  redisReply* _reply = (redisReply*)::redisCommand(_context, cmd.c_str());
  if (isError(_reply)) {
    freeReply(_reply);
    return false;
  }
  DEBUG << "_reply->type = " << _reply->type << std::endl;
  switch (_reply->type) {
    case REDIS_REPLY_ERROR: {
      freeReply(_reply);
      return false;
    } break;

    case REDIS_REPLY_ARRAY: {
      int32_t elements = _reply->elements;
      values.reserve(elements);
      for (int32_t i = 0; i < elements; ++i) {
        std::string strTemp(_reply->element[i]->str, _reply->element[i]->len);
        values.push_back(strTemp);
      }
      freeReply(_reply);
      return true;
    } break;

    case REDIS_REPLY_INTEGER: {
      std::string num = num2str<int64_t>(_reply->integer);
      values.push_back(num);
      freeReply(_reply);
      return true;
    } break;

    case REDIS_REPLY_STRING: {
      std::string tmp = _reply->str;
      values.push_back(tmp);
      freeReply(_reply);
      return true;
    } break;

    case REDIS_REPLY_STATUS: {
      std::string tmp = _reply->str;
      values.push_back(tmp);
      freeReply(_reply);
      return true;
    } break;

    case REDIS_REPLY_NIL: {
      freeReply(_reply);
      return true;
    } break;

    default: {
      freeReply(_reply);
      return false;
    }
  }
}

template <typename TypeHashKey>
void redis<TypeHashKey>::connect() {
  if (_connected) {
    DEBUG << "Redis connected!" << std::endl;
    return;
  }
  _connected = true;
  _context = ::redisConnect(_ip.c_str(), _port);
  if (_context && _context->err) {
    DEBUG << "Redis connection fail!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (_pwd.empty()) {
    std::vector<std::string> status;
    exec("PING", status);
    if ((!status.empty()) && strcasecmp(status[0].c_str(), "PONG") == 0) {
      return;
    } else {
      DEBUG << "Redis connection fail with pwd!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

template <typename TypeHashKey>
void redis<TypeHashKey>::disconnect() {
  if (_connected) {
    DEBUG << "free _context" << std::endl;
    ::redisFree(_context);
    _connected = false;
  }
}
template <typename TypeHashKey>
void redis<TypeHashKey>::look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
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
void redis<TypeHashKey>::load_data(const std::string& model_config_path) {
  return;
}

template <typename TypeHashKey>
void redis<TypeHashKey>::freeReply(redisReply* reply) {
  if (reply) {
    ::freeReplyObject(reply);
    reply = NULL;
  }
}

template <typename TypeHashKey>
bool redis<TypeHashKey>::mset(const std::vector<TypeHashKey>& keys, std::vector<float>& values,
                              const std::string modelname, const std::string tablename,
                              size_t embedding_size) {
  size_t num_key = 10000;
  size_t iter = keys.size() / num_key;
  if (_context == NULL) {
    std::cout << "redis local fail!" << std::endl;
    return false;
  }

  if (keys.size() != values.size() / embedding_size) {
    return false;
  }
  int len = 2 * num_key;
  int argc = 2;
  const char* argv[len + 2];
  size_t argvlen[len + 2];
  char argv1[] = "HMSET";
  argv[0] = argv1;
  argvlen[0] = strlen("HMSET");

  const std::string table = modelname + tablename;
  std::cout << "Local Redis is initializing the embedding table: " << table << std::endl;
  argv[1] = table.c_str();
  argvlen[1] = table.length();

  for (size_t j = 0; j < iter; j++) {
    size_t index = j * num_key;
    argc = 2;
    for (size_t t = 0; t < num_key; t++) {
      argvlen[argc] = sizeof(TypeHashKey);
      argv[argc] = reinterpret_cast<const char*>(&keys[t + index]);
      argc++;

      argvlen[argc] = sizeof(float) * embedding_size;
      argv[argc] = reinterpret_cast<const char*>(&(values[(index + t) * embedding_size]));
      argc++;
    }
    redisReply* _reply = (redisReply*)::redisCommandArgv(_context, argc, argv, argvlen);
    if (_reply && _reply->type == REDIS_REPLY_STATUS) {
      if (strcasecmp(_reply->str, "OK") == 0) {
        std::cout << "Iteration " << j << " insert successfully" << std::endl;
      }
    } else {
      std::cout << "Iteration " << j << " insert fail, Please check Redis Cluster Status!"
                << std::endl;
    }
  }
  if ((iter + 1) * num_key >= keys.size()) {
    size_t rest_keys = keys.size() - iter * num_key;
    size_t index = iter * num_key;
    argc = 2;
    for (size_t t = 0; t < rest_keys; t++) {
      argvlen[argc] = sizeof(TypeHashKey);
      argv[argc] = reinterpret_cast<const char*>(&keys[t + index]);
      argc++;

      argvlen[argc] = sizeof(float) * embedding_size;
      argv[argc] = reinterpret_cast<const char*>(&(values[(index + t) * embedding_size]));
      argc++;
    }
    redisReply* _reply = (redisReply*)::redisCommandArgv(_context, argc, argv, argvlen);
    if (_reply && _reply->type == REDIS_REPLY_STATUS) {
      if (strcasecmp(_reply->str, "OK") == 0) {
        std::cout << "The last iteration insert successfully" << std::endl;
        freeReply(_reply);
        return true;
      } else {
      }
    }
  }

  freeReply(_reply);

  return false;
}
template <typename TypeHashKey>
bool redis<TypeHashKey>::mget(const TypeHashKey* keys, std::vector<float>& values, size_t length,
                              std::string modelname, const std::string tablename,
                              size_t embedding_size) {
  if (length <= 0) {
    return true;
  }
  if (_context == NULL) {
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
  std::cout << "Redis gets missing keys num: " << length << std::endl;
  std::cout << "Redis gets missing keys from model: " << modelname << " and table: " << tablename
            << std::endl;
  argv[1] = table.c_str();
  argvlen[1] = table.length();

  for (size_t t = 0; t < len; t++) {
    argvlen[argc] = sizeof(TypeHashKey);
    argv[argc] = reinterpret_cast<const char*>(&keys[t]);
    argc++;
  }
  redisReply* _reply = (redisReply*)::redisCommandArgv(_context, argc, argv, argvlen);

  if (_reply && _reply->type == REDIS_REPLY_ARRAY) {
    values.clear();
    for (unsigned int i = 0; i < _reply->elements; i++) {
      std::string strTemp(_reply->element[i]->str, _reply->element[i]->len);
      if (strTemp.size() <= 0) {
        std::vector<float> default_emb_vec(embedding_size, 0);
        values.insert(values.end(), default_emb_vec.data(),
                      default_emb_vec.data() + embedding_size);

      } else {
        float* temp = (reinterpret_cast<float*>(_reply->element[i]->str));
        values.insert(values.end(), temp, temp + embedding_size);
      }
    }
  }

  freeReply(_reply);
  return true;
}

template <typename TypeHashKey>
bool redis<TypeHashKey>::cmset(const std::vector<TypeHashKey>& keys, std::vector<float>& values,
                               const std::string modelname, const std::string tablename,
                               size_t embedding_size, const float cache_size_percentage_redis) {
  r3c::CRedisClient rc_client(_ip);
  size_t num_key = 10000;
  size_t total_key = keys.size() * cache_size_percentage_redis;
  size_t iter = total_key / num_key;
  if (!rc_client.cluster_mode()) {
    std::cout << "Please checke the Redis Cluster status" << std::endl;
    return false;
  }
  if (keys.size() != values.size() / embedding_size) {
    return false;
  }
  const std::string table = modelname + tablename;
  size_t key_len = sizeof(TypeHashKey) + table.size();
  size_t prefix_len = table.size();
  size_t val_len = sizeof(float) * embedding_size;
  std::cout << "Reidis Cluster is initializing the embedding table: " << table << std::endl;
  char* buf = new char[key_len];
  char* val_buf = new char[val_len];
  for (size_t j = 0; j < iter; j++) {
    size_t index = j * num_key;
    for (size_t t = 0; t < num_key; t++) {
      std::string key;
      std::string value = "";
      if (keys[t + index] >= 0) {
        memcpy(buf, table.data(), prefix_len);
        memcpy(buf + prefix_len, reinterpret_cast<const char*>(&keys[t + index]),
               sizeof(TypeHashKey));
        memcpy(val_buf, reinterpret_cast<const char*>(&(values[(index + t) * embedding_size])),
               val_len);
        key.assign(buf, key_len);
        value.assign(val_buf, val_len);
        rc_client.set(key, value);
      }
    }
  }
  if ((iter + 1) * num_key >= total_key) {
    size_t rest_keys = total_key - iter * num_key;
    size_t index = iter * num_key;
    for (size_t t = 0; t < rest_keys; t++) {
      if (keys[t + index] >= 0) {
        std::string key;
        std::string value = "";
        memcpy(buf, table.data(), prefix_len);
        memcpy(buf + prefix_len, reinterpret_cast<const char*>(&keys[t + index]),
               sizeof(TypeHashKey));
        memcpy(val_buf, reinterpret_cast<const char*>(&(values[(index + t) * embedding_size])),
               val_len);
        key.assign(buf, key_len);
        value.assign(val_buf, val_len);
        rc_client.set(key, value);
      }
    }
    std::cout << "Last Iteration insert successfully" << std::endl;
  }
  delete buf;
  delete val_buf;
  return true;
}

template <class T>
T str2num(const std::string& str) {
  std::stringstream ss;
  T num;
  ss << str;
  ss >> num;
  return num;
}

template class redis<unsigned int>;
template class redis<long long>;
}  // namespace HugeCTR