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

#include "hps/plugin/facade.hpp"

namespace HierarchicalParameterServer {

Facade::Facade() : lookup_manager_(LookupManager::Create()) {}

Facade* Facade::instance() {
  static Facade instance;
  return &instance;
}

void Facade::operator delete(void*) {
  throw std::domain_error("This pointer cannot be manually deleted.");
}

void Facade::init(const char* ps_config_file, pluginType_t plugin_type, int32_t global_batch_size,
                  int32_t num_replicas_in_sync) {
  std::call_once(lookup_manager_init_once_flag_, [this, ps_config_file, plugin_type,
                                                  global_batch_size, num_replicas_in_sync]() {
    HugeCTR::parameter_server_config ps_config{ps_config_file};
    lookup_manager_->init(ps_config, plugin_type, global_batch_size, num_replicas_in_sync);
  });
}

void Facade::forward(const char* model_name, int32_t table_id, int32_t global_replica_id,
                     size_t num_keys, size_t emb_vec_size, const void* d_keys, void* d_vectors) {
  lookup_manager_->forward(std::string(model_name), table_id, global_replica_id, num_keys,
                           emb_vec_size, d_keys, d_vectors);
}

}  // namespace HierarchicalParameterServer