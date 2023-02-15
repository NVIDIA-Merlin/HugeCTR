/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/lookup_session.hpp>

namespace HierarchicalParameterServer {

using namespace HugeCTR;

typedef enum {
  TENSORFLOW = 0,
  TENSORRT = 1,
} pluginType_t;

class LookupManager final {
 public:
  ~LookupManager() = default;
  LookupManager(const LookupManager&) = delete;
  LookupManager& operator=(const LookupManager&) = delete;

  static std::shared_ptr<LookupManager> Create();

  // global_batch_size and num_replicas_in_sync only valid for TensorFlow plugin
  void init(parameter_server_config& ps_config, const pluginType_t plugin_type,
            const int32_t global_batch_size, const int32_t num_replicas_in_sync);

  // TODO: remove this method
  void forward(const std::string& model_name, const int32_t table_id,
               const int32_t global_replica_id, const size_t num_keys, const size_t emb_vec_size,
               const void* values_ptr, void* emb_vector_ptr);

  void forward(const std::string& model_name, const int32_t table_id,
               const int32_t global_replica_id, const size_t num_keys, const size_t emb_vec_size,
               const void* values_ptr, void* emb_vector_ptr, bool i64_input_tensor,
               cudaStream_t context_stream);

  void init_check(parameter_server_config& ps_config, const int32_t global_batch_size,
                  const int32_t num_replicas_in_sync) const;

  void forward_check(const std::string& model_name, int32_t table_id, int32_t global_replica_id,
                     size_t num_keys, size_t emb_vec_size, bool i64_input_tensor) const;

 private:
  LookupManager();
  bool initialized_;
  std::shared_ptr<HierParameterServerBase> parameter_server_;
  std::map<std::string, std::map<size_t, std::shared_ptr<LookupSessionBase>>> lookup_session_map_;
};

}  // namespace HierarchicalParameterServer
