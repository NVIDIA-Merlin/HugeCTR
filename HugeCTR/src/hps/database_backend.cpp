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
#include <hps/database_backend.hpp>
#include <sstream>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename TKey>
size_t DatabaseBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                       const TKey* keys,
                                       const std::chrono::microseconds& time_budget) const {
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                    const TKey* const keys, const DatabaseHitCallback& on_hit,
                                    const DatabaseMissCallback& on_miss,
                                    const std::chrono::microseconds& time_budget) {
  for (size_t i = 0; i < num_keys; i++) {
    on_miss(i);
  }
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                    const size_t* indices, const TKey* const keys,
                                    const DatabaseHitCallback& on_hit,
                                    const DatabaseMissCallback& on_miss,
                                    const std::chrono::microseconds& time_budget) {
  const size_t* const indices_end = &indices[num_indices];
  for (; indices != indices_end; indices++) {
    on_miss(*indices);
  }
  return 0;
}

template <typename TKey>
size_t DatabaseBackend<TKey>::evict(const std::vector<std::string>& table_names) {
  size_t n = 0;
  for (const std::string& table_name : table_names) {
    n += evict(table_name);
  }
  return n;
}

template class DatabaseBackend<unsigned int>;
template class DatabaseBackend<long long>;

DatabaseBackendError::DatabaseBackendError(const std::string& backend, const size_t partition,
                                           const std::string& what)
    : backend_{backend}, partition_{partition}, what_{what} {}

std::string DatabaseBackendError::to_string() const {
  std::ostringstream os;
  os << backend_ << " DB Backend error (partition = " << partition_ << "): " << what_;
  return os.str();
}

template <typename TKey>
VolatileBackend<TKey>::VolatileBackend(const size_t max_get_batch_size,
                                       const size_t max_set_batch_size,
                                       const size_t overflow_margin,
                                       const DatabaseOverflowPolicy_t overflow_policy,
                                       const double overflow_resolution_target)
    : max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size),
      overflow_margin_(overflow_margin),
      overflow_policy_(overflow_policy),
      overflow_resolution_target_(hctr_safe_cast<size_t>(
          static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5)) {
  HCTR_CHECK(overflow_resolution_target_ <= overflow_margin_);
}

template <typename TKey>
std::future<void> VolatileBackend<TKey>::insert_async(
    const std::string& table_name, const std::shared_ptr<std::vector<TKey>>& keys,
    const std::shared_ptr<std::vector<char>>& values, size_t value_size) {
  HCTR_CHECK(keys->size() * value_size == values->size());
  return background_worker_.submit([this, table_name, keys, values, value_size]() {
    this->insert(table_name, keys->size(), keys->data(), values->data(), value_size);
  });
}

template <typename TKey>
void VolatileBackend<TKey>::synchronize() {
  background_worker_.await_idle();
}

template class VolatileBackend<unsigned int>;
template class VolatileBackend<long long>;

template class PersistentBackend<unsigned int>;
template class PersistentBackend<long long>;

}  // namespace HugeCTR