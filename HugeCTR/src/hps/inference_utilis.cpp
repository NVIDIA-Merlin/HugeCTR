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

#include <hps/inference_utils.hpp>

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

}  // namespace HugeCTR
