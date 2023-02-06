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
#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <memory>
#include <string>
#include <vector>

namespace HugeCTR {

class HierParameterServerBase {
 public:
  virtual ~HierParameterServerBase() = 0;
  HierParameterServerBase() = default;
  HierParameterServerBase(HierParameterServerBase const&) = delete;
  HierParameterServerBase& operator=(HierParameterServerBase const&) = delete;

  static constexpr const char* PS_EMBEDDING_TABLE_TAG_PREFIX = "hps_et";

  static std::string make_tag_name(const std::string& model_name,
                                   const std::string& embedding_table_name,
                                   const bool check_arguments = true);

  static std::shared_ptr<HierParameterServerBase> create(
      const parameter_server_config& ps_config,
      std::vector<InferenceParams>& inference_params_array);

  static std::shared_ptr<HierParameterServerBase> create(const std::string& hps_json_config_file);

  virtual void update_database_per_model(const InferenceParams& inference_params) = 0;
  virtual void create_embedding_cache_per_model(InferenceParams& inference_params) = 0;
  virtual void init_ec(
      InferenceParams& inference_params,
      std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map) = 0;
  virtual void destory_embedding_cache_per_model(const std::string& model_name) = 0;
  virtual std::shared_ptr<EmbeddingCacheBase> get_embedding_cache(const std::string& model_name,
                                                                  int device_id) = 0;

  virtual void erase_model_from_hps(const std::string& model_name) = 0;

  virtual void* apply_buffer(const std::string& model_name, int device_id,
                             CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER) = 0;
  virtual void free_buffer(void* p) = 0;
  virtual void lookup(const void* h_keys, size_t length, float* h_vectors,
                      const std::string& model_name, size_t table_id) = 0;
  virtual void refresh_embedding_cache(const std::string& model_name, int device_id) = 0;
  virtual void insert_embedding_cache(size_t table_id,
                                      std::shared_ptr<EmbeddingCacheBase> embedding_cache,
                                      EmbeddingCacheWorkspace& workspace_handler,
                                      cudaStream_t stream) = 0;
  virtual void parse_hps_configuraion(const std::string& hps_json_config_file) = 0;
  virtual std::map<std::string, InferenceParams> get_hps_model_configuration_map() = 0;
  virtual void set_profiler(int interation, int warmup, bool enable_bench) = 0;
  virtual void profiler_print() = 0;
};

}  // namespace HugeCTR