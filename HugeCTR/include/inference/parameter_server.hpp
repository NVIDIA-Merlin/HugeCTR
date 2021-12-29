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
#include <inference/database_backend.hpp>
#include <inference/embedding_interface.hpp>
#include <inference/inference_utils.hpp>
#include <inference/memory_pool.hpp>
#include <inference/message.hpp>
#include <iostream>
#include <memory>
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
  virtual ~parameter_server_base();

  // Used in some backend implementations to name embedding tables appropriately.
  // Note: Kafka requires [a-zA-Z0-9\\._\\-]{1,249} for topic names. Also at other
  //       locations we may want to use this directly in regular expressions. So
  //       let's avoid using "." (dot).
  static constexpr const char* PS_EMBEDDING_TABLE_TAG_PREFIX = "hctr_et";

  static std::string make_tag_name(const std::string& model_name,
                                   const std::string& embedding_table);
};

template <typename TypeHashKey>
class parameter_server : public parameter_server_base, public HugectrUtility<TypeHashKey> {
 public:
  parameter_server(const std::string& framework_name,
                   const std::vector<std::string>& model_config_path,
                   std::vector<InferenceParams>& inference_params_array);
  virtual ~parameter_server();
  // Should not be called directly, should be called by embedding cache
  virtual void look_up(const TypeHashKey* h_embeddingcolumns, size_t length,
                       float* h_embeddingoutputvector, const std::string& model_name,
                       size_t embedding_table_id);
  virtual void* ApplyBuffer(const std::string& modelname, int deviceid,
                            CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER);
  virtual void FreeBuffer(void* p);
  virtual void refresh_embedding_cache(const std::string& model_name, int device_id);
  virtual void insert_embedding_cache(embedding_interface* embedding_cache,
                                      embedding_cache_config& cache_config,
                                      embedding_cache_workspace& workspace_handler,
                                      const std::vector<cudaStream_t>& streams);
  virtual void update_database_per_model(const std::string& model_config_path,
                                         const InferenceParams& inference_param);
  virtual std::shared_ptr<embedding_interface> GetEmbeddingCache(const std::string& modelname,
                                                                 int deviceid);

 private:
  // The framework name
  std::string framework_name_;
  // Currently, embedding tables are implemented as CPU hashtable, 1 hashtable per embedding table
  // per model
  // The parameter server configuration
  parameter_server_config ps_config_;

  // Database layers for multi-tier cache/lookup.
  std::shared_ptr<DatabaseBackend<TypeHashKey>> volatile_db_;
  double volatile_db_cache_rate_ = 0.0;
  std::shared_ptr<DatabaseBackend<TypeHashKey>> persistent_db_;

  std::vector<std::shared_ptr<DatabaseBackend<TypeHashKey>>> db_stack_;

  // Realtime data ingestion.
  std::unique_ptr<MessageSource<TypeHashKey>> volatile_db_source_;
  std::unique_ptr<MessageSource<TypeHashKey>> persistent_db_source_;
  inference_memory_pool_size_config memory_pool_config;

  std::shared_ptr<ManagerPool> bufferpool;
  std::map<std::string, std::map<int64_t, std::shared_ptr<embedding_interface>>> model_cache_map;

  void parse_networks_per_model_(const std::vector<std::string>& model_config_path,
                                 std::vector<InferenceParams>& inference_params_array);
};

}  // namespace HugeCTR
