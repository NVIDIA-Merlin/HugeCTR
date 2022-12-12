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

#include "hps/plugin/lookup_manager.hpp"

#include "base/debug/logger.hpp"

namespace HierarchicalParameterServer {

std::shared_ptr<LookupManager> LookupManager::Create() {
  return std::shared_ptr<LookupManager>(new LookupManager());
}

LookupManager::LookupManager() : initialized_{false} {}

void LookupManager::init(parameter_server_config& ps_config, pluginType_t plugin_type,
                         int32_t global_batch_size, int32_t num_replicas_in_sync) {
  initialized_ = true;
  for (auto& inference_params : ps_config.inference_params_array) {
    HCTR_CHECK_HINT(inference_params.use_gpu_embedding_cache,
                    "use_gpu_embedding_cache must be true for HPS plugin.");
  }
  // Check for different plugin types
  if (plugin_type == pluginType_t::TENSORFLOW) {
    check(ps_config, global_batch_size, num_replicas_in_sync);
  } else if (plugin_type == pluginType_t::TENSORRT) {
    for (auto& inference_params : ps_config.inference_params_array) {
      HCTR_CHECK_HINT(!inference_params.i64_input_key,
                      "i64_input_key must be false for HPS TensorRT plugin.");
    }
  }

  // Create the HPS for all models on all the deployed devices
  parameter_server_ = HierParameterServerBase::create(ps_config, ps_config.inference_params_array);

  // Initialie the resources for each model
  for (auto& inference_params : ps_config.inference_params_array) {
    // Create the lookup sessions on all the deployed devices
    std::map<size_t, std::shared_ptr<LookupSessionBase>> lookup_sessions;
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      auto embedding_cache = parameter_server_->get_embedding_cache(inference_params.model_name,
                                                                    inference_params.device_id);
      auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);
      lookup_sessions.emplace(device_id, lookup_session);
    }
    lookup_session_map_.emplace(inference_params.model_name, lookup_sessions);
  }
}

void LookupManager::forward(const std::string& model_name, int32_t table_id,
                            int32_t global_replica_id, size_t num_keys, size_t emb_vec_size,
                            const void* values_ptr, void* emb_vector_ptr) {
  if (!initialized_) {
    HCTR_LOG_S(ERROR, WORLD) << "HPS must be initialized before execution" << std::endl;
    return;
  }
  if (lookup_session_map_.find(model_name) == lookup_session_map_.end()) {
    HCTR_LOG_S(ERROR, WORLD) << "Cannot find the model " << model_name << " in HPS" << std::endl;
    return;
  }
  if (lookup_session_map_.find(model_name)->second.find(global_replica_id) ==
      lookup_session_map_.find(model_name)->second.end()) {
    HCTR_LOG_S(ERROR, WORLD) << "Model " << model_name << " is NOT deployed on the device "
                             << global_replica_id << std::endl;
    return;
  }

  auto lookup_session =
      lookup_session_map_.find(model_name)->second.find(global_replica_id)->second;
  auto inference_params = lookup_session->get_inference_params();
  size_t num_tables = inference_params.sparse_model_files.size();

  if (table_id < 0 || table_id >= num_tables) {
    HCTR_LOG_S(ERROR, WORLD) << "table_id for " << model_name << " should be from 0 to "
                             << num_tables - 1 << std::endl;
    return;
  }

  if (num_keys > inference_params.max_batchsize *
                     inference_params.maxnum_catfeature_query_per_table_per_sample[table_id]) {
    HCTR_LOG_S(ERROR, WORLD)
        << "num_keys must be no larger than "
        << inference_params.max_batchsize *
               inference_params.maxnum_catfeature_query_per_table_per_sample[table_id]
        << std::endl;
    return;
  }
  if (emb_vec_size != inference_params.embedding_vecsize_per_table[table_id]) {
    HCTR_LOG_S(ERROR, WORLD) << "emb_vec_size must be equal to "
                             << inference_params.embedding_vecsize_per_table[table_id] << std::endl;
    return;
  }
  lookup_session->lookup_from_device(values_ptr, reinterpret_cast<float*>(emb_vector_ptr), num_keys,
                                     table_id);
}

void LookupManager::check(parameter_server_config& ps_config, int32_t global_batch_size,
                          const int32_t num_replicas_in_sync) const {
  HCTR_CHECK_HINT(global_batch_size > 0, "global_batch_size must be > 0.");
  HCTR_CHECK_HINT(num_replicas_in_sync > 0, "num_replicas_in_sync must be > 0.");
  HCTR_CHECK_HINT(global_batch_size % num_replicas_in_sync == 0,
                  "global_batch_size must be divisible by num_replicas_in_sync.");
  size_t local_batch_size = global_batch_size / num_replicas_in_sync;

  for (auto& inference_params : ps_config.inference_params_array) {
    sort(inference_params.deployed_devices.begin(), inference_params.deployed_devices.end());
    auto check = [](const std::vector<int>& vec) {
      for (size_t i{0}; i < vec.size(); ++i) {
        if (vec[i] != i) return false;
      }
      return true;
    };
    HCTR_CHECK_HINT(inference_params.i64_input_key,
                    "i64_input_key must be true for HPS TensorFlow plugin.");
    HCTR_CHECK_HINT(
        inference_params.deployed_devices.size() == num_replicas_in_sync,
        "inference_params.deployed_devices.size() must be equal to num_replicas_in_sync.");
    HCTR_CHECK_HINT(check(inference_params.deployed_devices),
                    "inference_params.deployed_devices should contain exactly from 0 to "
                    "num_replicas_in_sync-1.");
    HCTR_CHECK_HINT(local_batch_size <= inference_params.max_batchsize,
                    "global_batch_size / num_replicas_in_sync must be <= max_batchsize configured "
                    "in ps_config.json.");
  }
}

}  // namespace HierarchicalParameterServer