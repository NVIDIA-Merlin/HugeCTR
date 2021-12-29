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

#include <base/debug/logger.hpp>
#include <inference/database_backend.hpp>
#include <sstream>

namespace HugeCTR {

template <typename TKey>
size_t DatabaseBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                       const TKey* keys) const {
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                    const TKey* const keys, char* const values,
                                    const size_t value_size,
                                    MissingKeyCallback& missing_callback) const {
  for (size_t i = 0; i < num_keys; i++) {
    missing_callback(i);
  }
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                    const size_t* indices, const TKey* const keys,
                                    char* const values, const size_t value_size,
                                    MissingKeyCallback& missing_callback) const {
  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; indices++) {
    missing_callback(*indices);
  }
  return 0;
}

template class DatabaseBackend<unsigned int>;
template class DatabaseBackend<long long>;

DatabaseBackendError::DatabaseBackendError(const std::string& backend, const size_t partition,
                                           const std::string& what)
    : backend_{backend}, partition_{partition}, what_{what} {}

std::string DatabaseBackendError::to_string() const {
  std::stringstream ss;
  ss << backend_ << " DB Backend error (partition = " << partition_ << "): " << what_;
  return ss.str();
}

template <typename TKey>
VolatileBackend<TKey>::VolatileBackend(const size_t overflow_margin,
                                       const DatabaseOverflowPolicy_t overflow_policy,
                                       const double overflow_resolution_target)
    : overflow_margin_(overflow_margin),
      overflow_policy_(overflow_policy),
      overflow_resolution_target_(hctr_safe_cast<size_t>(
          static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5)) {
  HCTR_CHECK(overflow_resolution_target_ <= overflow_margin_);
}

template class VolatileBackend<unsigned int>;
template class VolatileBackend<long long>;

template class PersistentBackend<unsigned int>;
template class PersistentBackend<long long>;

}  // namespace HugeCTR