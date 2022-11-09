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

#pragma once

#include <thrust/device_vector.h>

#include <cuco/detail/dynamic_map_kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys
 *
 * Automatically grows capacity as necessary until device memory runs out.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires keys that are Arithmetic
 * - Does not support erasing keys
 * - Capacity does not shrink automatically
 * - Requires the user to specify sentinel values for both key and mapped
 * value to indicate empty slots
 * - Does not support concurrent insert and find operations
 *
 * The `dynamic_map` supports host-side "bulk" operations which include
 * `insert`, `find` and `contains`. These are to be used when there are a
 * large number of keys to insert or lookup in the map. For example, given a
 * range of keys specified by device-accessible iterators, the bulk `insert`
 * function will insert all keys into the map.
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 */
template <typename Key, typename Element, typename Initializer>
class dynamic_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

 public:
  static auto constexpr Scope = cuda::thread_scope_device;

  using key_type = Key;
  using element_type = Element;
  using pair_type = cuco::pair_type<key_type, element_type *>;
  using const_pair_type = cuco::pair_type<key_type, element_type const *>;
  using view_type = typename static_map<key_type, element_type, Initializer>::device_view;
  using mutable_view_type =
      typename static_map<key_type, element_type, Initializer>::device_mutable_view;
  using atomic_ctr_type = cuda::atomic<size_t, Scope>;

  dynamic_map(dynamic_map const &) = delete;
  dynamic_map(dynamic_map &&) = delete;
  dynamic_map &operator=(dynamic_map const &) = delete;
  dynamic_map &operator=(dynamic_map &&) = delete;

  /**
   * @brief Construct a dynamically-sized map with the specified initial
   * capacity, growth factor and sentinel values.
   *
   * The capacity of the map will automatically increase as the user adds
   * key/value pairs using `insert`.
   *
   * Capacity increases by a factor of growth_factor each time the size of the
   * map exceeds a threshold occupancy. The performance of `find` and
   * `contains` decreases somewhat each time the map's capacity grows.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved
   * and undefined behavior results from attempting to insert any key/value
   * pair that contains either.
   *
   * @param initial_capacity The initial number of slots in the map
   */
  dynamic_map(uint32_t dimension, size_t initial_capacity, Initializer const &initializer);

  void initialize(cudaStream_t stream = 0);
  void uninitialize(cudaStream_t stream = 0);

  /**
   * @brief Grows the capacity of the map so there is enough space for `n`
   * key/value pairs.
   *
   * If there is already enough space for `n` key/value pairs, the capacity
   * remains the same.
   *
   * @param n The number of key value pairs for which there must be space
   * @param stream Stream used for executing the kernels
   */
  void reserve(size_t n, cudaStream_t stream = 0);

  template <typename Hash = cuco::detail::MurmurHash3_32<key_type>>
  void lookup(key_type const *keys, element_type *values, size_t num_keys, cudaStream_t stream = 0,
              Hash hash = Hash{});

  template <typename Hash = cuco::detail::MurmurHash3_32<key_type>>
  void lookup_unsafe(key_type const *keys, element_type **values, size_t num_keys,
                     cudaStream_t stream = 0, Hash hash = Hash{});

  template <typename Hash = cuco::detail::MurmurHash3_32<key_type>>
  void scatter_add(key_type const *keys, element_type const *values, size_t num_keys,
                   cudaStream_t stream = 0, Hash hash = Hash{});

  template <typename Hash = cuco::detail::MurmurHash3_32<key_type>>
  void scatter_update(key_type const *keys, element_type const *values, size_t num_keys,
                      cudaStream_t stream = 0, Hash hash = Hash{});

  template <typename Hash = cuco::detail::MurmurHash3_32<key_type>>
  void remove(key_type const *keys, size_t num_keys, cudaStream_t stream = 0, Hash hash = Hash{});

  void eXport(key_type *keys, element_type *values, size_t num_keys, cudaStream_t stream = 0);

  void clear(cudaStream_t stream = 0);

  /**
   * @brief Gets the current number of elements in the map
   *
   * @return The current number of elements in the map
   */
  size_t get_size() const noexcept {
    size_t size = 0;
    for (auto &submap : submaps_) {
      size += submap->get_size();
    }
    return size;
  }

  size_t get_occupied_size() const noexcept {
    size_t occupied_size = 0;
    for (auto &submap : submaps_) {
      occupied_size += submap->get_occupied_size();
    }
    return occupied_size;
  }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  size_t get_capacity() const noexcept { return capacity_ - initial_capacity_; }

 private:
  static constexpr size_t max_num_submaps = 128;
  const float max_load_factor_{};   ///< Max load factor before capacity growth
  const size_t min_insert_size_{};  ///< min remaining capacity of submap for insert
  const uint32_t dimension_{};
  const size_t initial_capacity_{};

  size_t capacity_{};  ///< Maximum number of keys that can be inserted

  std::vector<std::unique_ptr<static_map<key_type, element_type, Initializer>>>
      submaps_;                                    ///< vector of pointers to each submap
  thrust::device_vector<view_type> submap_views_;  ///< vector of device views for each submap
  thrust::device_vector<mutable_view_type>
      submap_mutable_views_;  ///< vector of mutable device views for each submap
  atomic_ctr_type *occupied_size_per_submap_;
  atomic_ctr_type *reclaimed_size_per_submap_;
  atomic_ctr_type *num_successes_;
  atomic_ctr_type *h_occupied_size_per_submap_;
  atomic_ctr_type *h_reclaimed_size_per_submap_;
  atomic_ctr_type *h_num_successes_;

  Initializer initializer_{};
};
}  // namespace cuco

#include <cuco/detail/dynamic_map.inl>
