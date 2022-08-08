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

#include "lookup_manager.h"

#include "base/debug/logger.hpp"

namespace HierarchicalParameterServer {

std::shared_ptr<LookupManager> LookupManager::Create() {
  return std::shared_ptr<LookupManager>(new LookupManager());
}

LookupManager::LookupManager() : initialized_{false} {}

void LookupManager::init(parameter_server_config& ps_config, int32_t global_batch_size,
                         int32_t num_replicas_in_sync) {
  initialized_ = true;
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
    HCTR_CHECK_HINT(inference_params.i64_input_key, "inference_params.i64_input_key must be true.");
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

    // Allocate the host buffer per table per replica to support concurrent query
    std::map<size_t, std::vector<std::shared_ptr<void>>> h_values;
    for (const auto& device_id : inference_params.deployed_devices) {
      std::vector<std::shared_ptr<void>> h_values_per_table;
      for (size_t table_id = 0; table_id < inference_params.embedding_vecsize_per_table.size();
           ++table_id) {
        size_t capacity = inference_params.max_batchsize *
                          inference_params.maxnum_catfeature_query_per_table_per_sample[table_id];
        h_values_per_table.emplace_back(std::shared_ptr<size_t>(new size_t[capacity]));
      }
      h_values.emplace(device_id, h_values_per_table);
    }
    h_values_map_.emplace(inference_params.model_name, h_values);
  }
}

void LookupManager::forward(const std::string& model_name, int32_t table_id,
                            int32_t global_replica_id, size_t num_keys, size_t emb_vec_size,
                            const void* values_ptr, void* emb_vector_ptr) {
  HCTR_CHECK_HINT(initialized_,
                  "hierarchical_parameter_server.Init must be called before execution");
  HCTR_CHECK_HINT(lookup_session_map_.find(model_name) != lookup_session_map_.end(),
                  "Cannot find the model with the name %s in HPS", model_name.c_str());

  auto lookup_session =
      lookup_session_map_.find(model_name)->second.find(global_replica_id)->second;
  auto inference_params = lookup_session->get_inference_params();
  size_t num_tables = inference_params.sparse_model_files.size();

  HCTR_CHECK_HINT(table_id >= 0 && table_id < num_tables,
                  "table_id for %s should be from 0 to %lu, got: %d", model_name.c_str(),
                  num_tables - 1, table_id);

  HCTR_CHECK_HINT(
      num_keys <= inference_params.max_batchsize *
                      inference_params.maxnum_catfeature_query_per_table_per_sample[table_id],
      "num_keys must be <= inference_params.max_batchsize * "
      "inference_params.maxnum_catfeature_query_per_table_per_sample[table_id], but %lu > %lu * "
      "%lu",
      num_keys, inference_params.max_batchsize,
      inference_params.maxnum_catfeature_query_per_table_per_sample[table_id]);
  HCTR_CHECK_HINT(emb_vec_size == inference_params.embedding_vecsize_per_table[table_id],
                  "emb_vec_size must be equal to "
                  "inference_params.embedding_vecsize_per_table[table_id], but %lu != %lu",
                  emb_vec_size, inference_params.embedding_vecsize_per_table[table_id]);

  void* h_values =
      h_values_map_.find(model_name)->second.find(global_replica_id)->second[table_id].get();
  cudaMemcpy(h_values, values_ptr, num_keys * sizeof(size_t), cudaMemcpyDeviceToHost);
  lookup_session->lookup(reinterpret_cast<void*>(h_values),
                         reinterpret_cast<float*>(emb_vector_ptr), num_keys, table_id);
}

}  // namespace HierarchicalParameterServer