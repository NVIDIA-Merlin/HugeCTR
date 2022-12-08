/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace cuco {
namespace detail {

/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k` and
 * `v`.
 *
 * Each space in `slots` that can hold a key value pair is initialized to a
 * `pair_atomic_type` containing the key `k` and the value `v`.
 *
 * @tparam atomic_key_type Type of the `Key` atomic container
 * @tparam atomic_mapped_type Type of the `Value` atomic container
 * @tparam Key key type
 * @tparam Value value type
 * @tparam pair_atomic_type key/value pair type
 * @param slots Pointer to flat storage for the map's key/value pairs
 * @param k Key to which all keys in `slots` are initialized
 * @param v Value to which all values in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template <typename atomic_key_type, typename element_type, typename atomic_flag_type,
          typename pair_atomic_type, typename Key>
__global__ void initialize(pair_atomic_type *const slots, element_type *const elements, Key k,
                           uint32_t dimension, size_t size) {
  auto grid = cooperative_groups::this_grid();

  for (auto tid = grid.thread_rank(); tid < size; tid += grid.size()) {
    new (&slots[tid].first) atomic_key_type{k};
    new (&slots[tid].second) atomic_flag_type{};
  }

  for (auto tid = grid.thread_rank(); tid < dimension * size; tid += grid.size()) {
    elements[tid] = element_type{};
  }
}

}  // namespace detail
}  // namespace cuco
