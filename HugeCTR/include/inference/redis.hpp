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
#include <hiredis/hiredis.h>
#include <inference/redis_cluster.h>
#include <string.h>  //strcmpcase

#include <cstdlib>
#include <inference/database.hpp>
#include <inference/inference_utils.hpp>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class redis : public DataBase<TypeHashKey> {
 public:
  redis();
  redis(redisContext* context);
  redis(parameter_server_config ps_config);
  virtual ~redis();

  void init(const std::string& ip, const std::string& pwd);
  void look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
               float* h_embeddingoutputvector, const std::string& model_name,
               size_t embedding_table_id);

  void connect();
  void disconnect();

  void load_data(const std::string& model_config_path);
  bool mset(const std::vector<TypeHashKey>& keys, std::vector<float>& values,
            const std::string modelname, const std::string tablename, size_t embedding_size);
  bool mget(const TypeHashKey* keys, std::vector<float>& values, size_t length,
            const std::string modelname, const std::string tablename, size_t embedding_size);
  bool cmset(const std::vector<TypeHashKey>& keys, std::vector<float>& values,
             const std::string modelname, const std::string tablename, size_t embedding_size,
             const float cache_size_percentage_redis);
  bool exec(const std::string& cmd);
  bool exec(const std::string& cmd, std::vector<std::string>& values);
  redisReply* execargv(int argc, const char** argv, const size_t* argvlen);

  template <class T>
  T str2num(const std::string& str) {
    std::stringstream ss;
    T num;
    ss << str;
    ss >> num;
    return num;
  }

  template <class T>
  std::string num2str(const T& num) {
    std::stringstream ss;
    ss << num;
    std::string str;
    ss >> str;
    return str;
  }

 private:
  void freeReply(redisReply* reply);
  bool isError(redisReply* reply);

 private:
  // The Data Base name
  std::string data_base_name;
  // The parameter server configuration
  parameter_server_config ps_config_;
  redisContext* _context;
  // CRedisClient c_redis;
  redisReply* _reply;
  std::string _ip;
  int32_t _port;
  std::string _pwd;
  bool _connected;
};

}  // namespace HugeCTR