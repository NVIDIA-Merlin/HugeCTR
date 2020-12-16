/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <iostream>
#include <embedding.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <utils.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include <inference/hugectrmodel.hpp>
#include <inference/inference_utils.hpp>

namespace HugeCTR {

struct parameter_server_config{
  std::string emb_file_name_; // For multiple embedding tables, string vector is needed
  bool distributed_emb_; // true for distributed, false for localized, for multiple embedding tables, bool vector is needed
  size_t embedding_vec_size_; // For multiple embedding tables, size_t vector is needed
  float default_emb_vec_value_; // Used when embedding id cannot be found in embedding table
  size_t max_query_length_;
};

template <typename TypeHashKey>
class parameter_server : public HugectrUtility<TypeHashKey> {
 public:
  parameter_server(const std::string& model_name, const nlohmann::json& model_config);
  virtual ~parameter_server();

  void look_up(const TypeHashKey* embeddingcolumns, size_t length, float* embeddingoutputvector, cudaStream_t stream);

 private:
  // The model name
  std::string model_name_;
  // The embedding table stored on CPU
  std::unordered_map<TypeHashKey, std::vector<float>> cpu_embedding_table_;
  // The temp buffers used internally
  TypeHashKey* h_embeddingcolumns_;
  float* h_embeddingoutputvector_;
  // The model configuration
  parameter_server_config model_config_;
};

}  // namespace HugeCTR