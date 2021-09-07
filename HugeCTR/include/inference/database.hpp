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
#include <inference/inference_utils.hpp>
#include <string>
#include <thread>
#include <utility>

namespace HugeCTR {

template <typename TypeHashKey>
class DataBase {
 public:
  DataBase();
  virtual ~DataBase() = 0;

  virtual void look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
                       float* h_embeddingoutputvector, const std::string& model_name,
                       size_t embedding_table_id) = 0;
  static DataBase* load_base(DATABASE_TYPE type, parameter_server_config ps_config);
  static DataBase* get_base(const std::string& db_type);

 private:
  // The Data Base name
  std::string framework_name_;
  // The parameter server configuration
  parameter_server_config ps_config_;
};

}  // namespace HugeCTR