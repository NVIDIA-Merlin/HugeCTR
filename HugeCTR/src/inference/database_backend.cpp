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

#include <inference/database_backend.hpp>

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

}  // namespace HugeCTR