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
#include <hps/database_backend.hpp>
#include <hps/embedding_cache_base.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/inference_utils.hpp>
#include <hps/memory_pool.hpp>
#include <hps/message.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class HierParameterServer : public HierParameterServerBase {
 public:
  virtual ~HierParameterServer();
  HierParameterServer(const parameter_server_config& ps_config,
                      std::vector<InferenceParams>& inference_params_array);
  HierParameterServer(HierParameterServer const&) = delete;
  HierParameterServer& operator=(HierParameterServer const&) = delete;

  virtual void update_database_per_model(const InferenceParams& inference_params);
  virtual void create_embedding_cache_per_model(InferenceParams& inference_params);
  virtual void init_ec(InferenceParams& inference_params,
                       std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map);
  virtual void destory_embedding_cache_per_model(const std::string& model_name);
  virtual std::shared_ptr<EmbeddingCacheBase> get_embedding_cache(const std::string& model_name,
                                                                  int device_id);

  virtual void erase_model_from_hps(const std::string& model_name);

  virtual void* apply_buffer(const std::string& model_name, int device_id,
                             CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER);
  virtual void free_buffer(void* p);
  virtual void lookup(const void* h_keys, size_t length, float* h_vectors,
                      const std::string& model_name, size_t table_id);
  virtual void refresh_embedding_cache(const std::string& model_name, int device_id);
  virtual void insert_embedding_cache(size_t table_id,
                                      std::shared_ptr<EmbeddingCacheBase> embedding_cache,
                                      EmbeddingCacheWorkspace& workspace_handler,
                                      cudaStream_t stream);
  virtual void parse_hps_configuraion(const std::string& hps_json_config_file);
  virtual std::map<std::string, InferenceParams> get_hps_model_configuration_map();

 private:
  // Parameter server configuration
  parameter_server_config ps_config_;
  // Database layers for multi-tier cache/lookup.
  std::unique_ptr<VolatileBackend<TypeHashKey>> volatile_db_;
  double volatile_db_cache_rate_;
  bool volatile_db_cache_missed_embeddings_;
  std::unique_ptr<PersistentBackend<TypeHashKey>> persistent_db_;
  // Realtime data ingestion.
  std::unique_ptr<MessageSource<TypeHashKey>> volatile_db_source_;
  std::unique_ptr<MessageSource<TypeHashKey>> persistent_db_source_;
  // Buffer pool that manages workspace and refreshspace of embedding caches
  std::shared_ptr<ManagerPool> buffer_pool_;
  // Configurations for memory pool
  inference_memory_pool_size_config memory_pool_config_;
  // Embedding caches of all models deployed on all devices, e.g., {"dcn": {0: dcn_embedding_cache0,
  // 1: dcnembedding_cache1}}
  std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>> model_cache_map_;
  // model configuration of all models deployed on HPS, e.g., {"dcn": dcn_inferenceParamesStruct}
  std::map<std::string, InferenceParams> inference_params_map_;
};

}  // namespace HugeCTR