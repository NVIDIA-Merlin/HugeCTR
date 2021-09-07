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

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

namespace HugeCTR {

template <typename TypeHashKey>
class rocks_db : public DataBase<TypeHashKey> {
 public:
  rocks_db();
  rocks_db(rocksdb::DB* db);
  rocks_db(parameter_server_config ps_config);
  virtual ~rocks_db();

  void init(const std::string& KDBPath);
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

 private:
  void freeReply();
  bool isError(rocksdb::Status s);

 private:
  // The Data Base name
  std::string data_base_name = "rocksdb";
  rocksdb::DB* _db;
  rocksdb::Options _options;
  // The parameter server configuration
  parameter_server_config ps_config_;
  std::string _kdbpath;
  bool _connected;
};

}  // namespace HugeCTR