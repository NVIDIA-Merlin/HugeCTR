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

#include <common.hpp>
#include <hps/inference_utils.hpp>
#include <nlohmann/json.hpp>
#include <parser.hpp>
#include <utils.hpp>

namespace HugeCTR {

std::ostream& operator<<(std::ostream& os, const DatabaseType_t value) {
  return os << hctr_enum_to_c_str(value);
}
std::ostream& operator<<(std::ostream& os, const DatabaseHashMapAlgorithm_t value) {
  return os << hctr_enum_to_c_str(value);
}
std::ostream& operator<<(std::ostream& os, const DatabaseOverflowPolicy_t value) {
  return os << hctr_enum_to_c_str(value);
}
std::ostream& operator<<(std::ostream& os, const UpdateSourceType_t value) {
  return os << hctr_enum_to_c_str(value);
}

std::optional<size_t> parameter_server_config::find_model_id(const std::string& model_name) const {
  const auto it = model_name_id_map_.find(model_name);
  if (it != model_name_id_map_.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

bool VolatileDatabaseParams::operator==(const VolatileDatabaseParams& p) const {
  return type == p.type &&
         // Backend specific.
         address == p.address && user_name == p.user_name && password == p.password &&
         algorithm == p.algorithm && num_partitions == p.num_partitions &&
         max_get_batch_size == p.max_get_batch_size && max_set_batch_size == p.max_set_batch_size &&
         // Overflow handling related.
         refresh_time_after_fetch == p.refresh_time_after_fetch &&
         overflow_margin == p.overflow_margin && overflow_policy == p.overflow_policy &&
         overflow_resolution_target == p.overflow_resolution_target &&
         // Caching behavior related.
         initial_cache_rate == p.initial_cache_rate &&
         cache_missed_embeddings == p.cache_missed_embeddings &&
         // Real-time update mechanism related.
         update_filters == p.update_filters;
}
bool VolatileDatabaseParams::operator!=(const VolatileDatabaseParams& p) const {
  return !operator==(p);
}

bool PersistentDatabaseParams::operator==(const PersistentDatabaseParams& p) const {
  return type == p.type &&
         // Backend specific.
         path == p.path && num_threads == p.num_threads && read_only == p.read_only &&
         max_get_batch_size == p.max_get_batch_size && max_set_batch_size == p.max_set_batch_size &&
         // Real-time update mechanism related.
         update_filters == p.update_filters;
}
bool PersistentDatabaseParams::operator!=(const PersistentDatabaseParams& p) const {
  return !operator==(p);
}

bool UpdateSourceParams::operator==(const UpdateSourceParams& p) const {
  return type == p.type &&
         // Backend specific.
         brokers == p.brokers && poll_timeout_ms == p.poll_timeout_ms &&
         max_receive_buffer_size == p.max_receive_buffer_size &&
         max_batch_size == p.max_batch_size && failure_backoff_ms == p.failure_backoff_ms;
}
bool UpdateSourceParams::operator!=(const UpdateSourceParams& p) const { return !operator==(p); }

VolatileDatabaseParams::VolatileDatabaseParams(
    const DatabaseType_t type,
    // Backend specific.
    const std::string& address, const std::string& user_name, const std::string& password,
    const DatabaseHashMapAlgorithm_t algorithm, const size_t num_partitions,
    const size_t max_get_batch_size, const size_t max_set_batch_size,
    // Overflow handling related.
    const bool refresh_time_after_fetch, const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy, const double overflow_resolution_target,
    // Caching behavior related.
    const double initial_cache_rate, const bool cache_missed_embeddings,
    // Real-time update mechanism related.
    const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      address(address),
      user_name(user_name),
      password(password),
      algorithm(algorithm),
      num_partitions(num_partitions),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Overflow handling related.
      refresh_time_after_fetch(refresh_time_after_fetch),
      overflow_margin(overflow_margin),
      overflow_policy(overflow_policy),
      overflow_resolution_target(overflow_resolution_target),
      // Caching behavior related.
      initial_cache_rate(initial_cache_rate),
      cache_missed_embeddings(cache_missed_embeddings),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

PersistentDatabaseParams::PersistentDatabaseParams(const DatabaseType_t type,
                                                   // Backend specific.
                                                   const std::string& path,
                                                   const size_t num_threads, const bool read_only,
                                                   const size_t max_get_batch_size,
                                                   const size_t max_set_batch_size,
                                                   // Real-time update mechanism related.
                                                   const std::vector<std::string>& update_filters)
    : type(type),
      // Backend specific.
      path(path),
      num_threads(num_threads),
      read_only(read_only),
      max_get_batch_size(max_get_batch_size),
      max_set_batch_size(max_set_batch_size),
      // Real-time update mechanism related.
      update_filters(update_filters) {}

UpdateSourceParams::UpdateSourceParams(const UpdateSourceType_t type,
                                       // Backend specific.
                                       const std::string& brokers, const size_t poll_timeout_ms,
                                       const size_t max_receive_buffer_size,
                                       const size_t max_batch_size, const size_t failure_backoff_ms)
    : type(type),
      // Backend specific.
      brokers(brokers),
      poll_timeout_ms(poll_timeout_ms),
      max_receive_buffer_size(max_receive_buffer_size),
      max_batch_size(max_batch_size),
      failure_backoff_ms(failure_backoff_ms) {}

InferenceParams::InferenceParams(
    const std::string& model_name, const size_t max_batchsize, const float hit_rate_threshold,
    const std::string& dense_model_file, const std::vector<std::string>& sparse_model_files,
    const int device_id, const bool use_gpu_embedding_cache, const float cache_size_percentage,
    const bool i64_input_key, const bool use_mixed_precision, const float scaler,
    const bool use_algorithm_search, const bool use_cuda_graph,
    const int number_of_worker_buffers_in_pool, const int number_of_refresh_buffers_in_pool,
    const float cache_refresh_percentage_per_iteration, const std::vector<int>& deployed_devices,
    const std::vector<float>& default_value_for_each_table,
    // Database backend.
    const VolatileDatabaseParams& volatile_db, const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source)
    : model_name(model_name),
      max_batchsize(max_batchsize),
      hit_rate_threshold(hit_rate_threshold),
      dense_model_file(dense_model_file),
      sparse_model_files(sparse_model_files),
      device_id(device_id),
      use_gpu_embedding_cache(use_gpu_embedding_cache),
      cache_size_percentage(cache_size_percentage),
      i64_input_key(i64_input_key),
      use_mixed_precision(use_mixed_precision),
      scaler(scaler),
      use_algorithm_search(use_algorithm_search),
      use_cuda_graph(use_cuda_graph),
      number_of_worker_buffers_in_pool(number_of_worker_buffers_in_pool),
      number_of_refresh_buffers_in_pool(number_of_refresh_buffers_in_pool),
      cache_refresh_percentage_per_iteration(cache_refresh_percentage_per_iteration),
      deployed_devices(deployed_devices),
      default_value_for_each_table(default_value_for_each_table),
      // Database backend.
      volatile_db(volatile_db),
      persistent_db(persistent_db),
      update_source(update_source) {}

parameter_server_config::parameter_server_config(
    const std::vector<std::string>& model_config_path_array,
    const std::vector<InferenceParams>& inference_params_array) {
  if (model_config_path_array.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The size of model_config_path_array and inference_params_array "
                   "are not consistent.");
  }
  for (size_t i = 0; i < model_config_path_array.size(); i++) {
    const auto& model_config_path = model_config_path_array[i];
    const auto& inference_params = inference_params_array[i];
    // Initialize <model_name, id> map
    if (model_name_id_map_.count(inference_params.model_name) == 0) {
      model_name_id_map_.emplace(inference_params.model_name, (size_t)model_name_id_map_.size());
    }

    // Initialize for each model
    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path));

    // Read inference config
    std::vector<std::string> emb_file_path;
    for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
      emb_file_path.emplace_back(inference_params.sparse_model_files[j]);
    }

    emb_file_name_[inference_params.model_name] = (emb_file_path);

    // Read embedding layer config
    std::vector<std::string> emb_table_name;
    std::vector<size_t> embedding_vec_size;
    std::vector<size_t> max_feature_num_per_sample_per_emb_table;
    std::vector<bool> distributed_emb;
    std::vector<float> default_emb_vec_value;

    // Search for all embedding layers
    const nlohmann::json& layers = get_json(model_config, "layers");
    for (size_t j = 0; j < layers.size(); j++) {
      const nlohmann::json& layer = layers[j];
      std::string layer_type = get_value_from_json<std::string>(layer, "type");
      if (layer_type.compare("Data") == 0) {
        const nlohmann::json& sparse_inputs = get_json(layer, "sparse");
        for (size_t k = 0; k < sparse_inputs.size(); k++) {
          max_feature_num_per_sample_per_emb_table.push_back(
              get_max_feature_num_per_sample_from_nnz_per_slot(sparse_inputs[k]));
        }
      } else if (layer_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
        distributed_emb.emplace_back(true);
        // parse embedding table name from network json file
        emb_table_name.emplace_back(get_value_from_json<std::string>(layer, "top"));
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else if (layer_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
                 layer_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
        distributed_emb.emplace_back(false);
        emb_table_name.emplace_back(get_value_from_json<std::string>(layer, "top"));
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(
            get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      } else {
        break;
      }
    }
    emb_table_name_[inference_params.model_name] = emb_table_name;
    embedding_vec_size_[inference_params.model_name] = embedding_vec_size;
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table;
    distributed_emb_.emplace_back(distributed_emb);
    default_emb_vec_value_.emplace_back(inference_params.default_value_for_each_table);
  }  // end for
}

}  // namespace HugeCTR
