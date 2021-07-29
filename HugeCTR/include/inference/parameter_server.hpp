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
#include <common.hpp>
#include <embedding.hpp>
#include <inference/inference_utils.hpp>
#include <iostream>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

class parameter_server_base {
 public:
  virtual ~parameter_server_base() = 0;
};

template <typename TypeHashKey>
class parameter_server : public parameter_server_base, public HugectrUtility<TypeHashKey> {
 public:
  parameter_server(const std::string& framework_name,
                   const std::vector<std::string>& model_config_path,
                   const std::vector<InferenceParams>& inference_params_array);
  virtual ~parameter_server();
  // Should not be called directly, should be called by embedding cache
  virtual void look_up(const TypeHashKey* h_embeddingcolumns, size_t length,
                       float* h_embeddingoutputvector, const std::string& model_name,
                       size_t embedding_table_id);

 private:
  // The framework name
  std::string framework_name_;
  // Currently, embedding tables are implemented as CPU hashtable, 1 hashtable per embedding table
  // per model
  std::vector<std::vector<std::unordered_map<TypeHashKey, std::vector<float>>>>
      cpu_embedding_table_;
  // The parameter server configuration
  parameter_server_config ps_config_;
};

}  // namespace HugeCTR
