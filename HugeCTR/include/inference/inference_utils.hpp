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
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace HugeCTR {
enum INFER_TYPE { TRITON, OTHER };
enum DATABASE_TYPE { LOCAL, REDIS, ROCKSDB, HIERARCHY };

struct InferenceParams {
  std::string model_name;
  size_t max_batchsize;
  float hit_rate_threshold;
  std::string dense_model_file;
  std::vector<std::string> sparse_model_files;
  int device_id;
  bool use_gpu_embedding_cache;
  float cache_size_percentage;
  bool i64_input_key;
  bool use_mixed_precision;
  float scaler;
  bool use_algorithm_search;
  bool use_cuda_graph;
  DATABASE_TYPE db_type;
  std::string redis_ip;
  std::string rocksdb_path;
  float cache_size_percentage_redis;
  InferenceParams(const std::string& model_name, const size_t max_batchsize,
                  const float hit_rate_threshold, const std::string& dense_model_file,
                  const std::vector<std::string>& sparse_model_files, const int device_id,
                  const bool use_gpu_embedding_cache, const float cache_size_percentage,
                  const bool i64_input_key, const bool use_mixed_precision = false,
                  const float scaler = 1.0, const bool use_algorithm_search = true,
                  const bool use_cuda_graph = true, DATABASE_TYPE db_type = DATABASE_TYPE::LOCAL,
                  const std::string redis_ip = "127.0.0.1:7000",
                  const std::string rocksdb_path = "",
                  const float cache_size_percentage_redis = 0.5);
};

struct parameter_server_config {
  std::map<std::string, size_t> model_name_id_map_;
  // Each vector should have size of M(# of models), where each element in the vector should be a
  // vector with size E(# of embedding tables in that model)
  std::vector<std::vector<std::string>>
      emb_file_name_;  // The file name per embedding table per model
  std::vector<std::vector<bool>>
      distributed_emb_;  // The file format flag per embedding table per model
  std::vector<std::vector<size_t>>
      embedding_vec_size_;  // The emb_vec_size per embedding table per model
  std::vector<std::vector<float>>
      default_emb_vec_value_;  // The defualt emb_vec value when emb_id cannot be found, per
                               // embedding table per model
};

// Base interface class for parameter_server
// 1 instance per HugeCTR backend(1 instance per all models per all embedding tables)
template <typename TypeHashKey>
class HugectrUtility {
 public:
  HugectrUtility();
  virtual ~HugectrUtility();
  // Should not be called directly, should be called by embedding cache
  virtual void look_up(const TypeHashKey* h_embeddingcolumns, size_t length,
                       float* h_embeddingoutputvector, const std::string& model_name,
                       size_t embedding_table_id) = 0;
  static HugectrUtility<TypeHashKey>* Create_Parameter_Server(
      INFER_TYPE Infer_type, const std::vector<std::string>& model_config_path,
      const std::vector<InferenceParams>& inference_params_array);
};

}  // namespace HugeCTR
