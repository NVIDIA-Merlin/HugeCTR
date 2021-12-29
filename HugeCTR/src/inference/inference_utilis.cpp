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

#include <inference/inference_utils.hpp>

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
         overflow_margin == p.overflow_margin && overflow_policy == p.overflow_policy &&
         overflow_resolution_target == p.overflow_resolution_target &&
         // Initialization related.
         initial_cache_rate == p.initial_cache_rate &&
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

}  // namespace HugeCTR
