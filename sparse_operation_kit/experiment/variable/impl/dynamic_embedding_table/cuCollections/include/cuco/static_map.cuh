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

// clang-format off
#include <cuda/std/atomic>
// clang-format on

#include <cuco/detail/error.hpp>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/insert_result.cuh>
#include <cuco/detail/iterator.cuh>
#include <cuco/detail/lock.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/detail/static_map_kernels.cuh>

namespace cuco {

template <typename Key, typename Element, typename Initializer>
class dynamic_map;

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations from
 * threads in device code. Concurrent insert and find are supported only if the
 * pair type is packable (see `cuco::detail::is_packable` constexpr).
 *
 * Current limitations:
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 * to indicate empty slots
 * - Conditionally support concurrent insert and find operations
 *
 * The `static_map` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `find`, and `contains`. These
 * APIs should be used when there are a large number of keys to insert or lookup
 * in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to perform
 * independent insert or find/contains operations from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `find` or
 * `contains`. The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Element Type of the mapped values
 */
template <typename Key, typename Element, typename Initializer>
class static_map {
  friend class dynamic_map<Key, Element, Initializer>;

  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

 public:
  static auto constexpr Scope = cuda::thread_scope_device;

  using key_type = Key;
  using element_type = Element;
  using pair_type = cuco::pair_type<key_type, element_type *>;
  using pointer_pair_type = cuco::pair_type<key_type, element_type **>;
  using const_pair_type = cuco::pair_type<key_type, element_type const *>;
  using atomic_key_type = cuda::atomic<key_type, Scope>;
  using lock_type = cuco::lock<Scope>;
  using pair_atomic_type = cuco::pair_type<atomic_key_type, lock_type>;
  using atomic_ctr_type = cuda::atomic<size_t, Scope>;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  static_assert(atomic_key_type::is_always_lock_free,
                "A key type larger than 8B is supported for only sm_70 and up.");
#endif

  static constexpr key_type empty_key_sentinel = ~static_cast<key_type>(0);
  static constexpr key_type reclaimed_key_sentinel = (~static_cast<key_type>(0)) ^ 1;

  static_map(static_map const &) = delete;
  static_map(static_map &&) = delete;
  static_map &operator=(static_map const &) = delete;
  static_map &operator=(static_map &&) = delete;

  /**
   * @brief Construct a fixed-size map with the specified capacity and sentinel values.
   * @brief Construct a statically sized map with the specified number of slots and sentinel values.
   *
   * The capacity of the map is fixed. Insert operations will not automatically
   * grow the map. Attempting to insert equal to or more unique keys than the
   * capacity of the map results in undefined behavior (there should be at least
   * one empty slot).
   *
   * Performance begins to degrade significantly beyond a load factor of ~70%.
   * For best performance, choose a capacity that will keep the load factor
   * below 70%. E.g., if inserting `N` unique keys, choose a capacity of
   * `N * (1/0.7)`.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved and
   * undefined behavior results from attempting to insert any key/value pair
   * that contains either.
   *
   * @param capacity The total number of slots in the map
   */
  static_map(uint32_t dimension, size_t capacity, atomic_ctr_type *occupied_size,
             atomic_ctr_type *reclaimed_size, atomic_ctr_type *h_occupied_size,
             atomic_ctr_type *h_reclaimed_size, Initializer initializer);

  void initialize(cudaStream_t stream = 0);
  void uninitialize(cudaStream_t stream = 0);

 private:
  class device_view_base {
   protected:
    // Import member type definitions from `static_map`
    using key_type = key_type;
    using element_type = element_type;
    using iterator = cuco::iterator<pair_atomic_type, element_type>;
    using const_iterator = cuco::const_iterator<pair_atomic_type, element_type>;

   private:
    uint32_t dimension_{};
    size_t capacity_{};          ///< Total number of slots
    pair_atomic_type *slots_{};  ///< Pointer to flat slots storage
    element_type *elements_{};
    Initializer initializer_{};

   protected:
    __host__ __device__ device_view_base(uint32_t dimension, size_t capacity,
                                         pair_atomic_type *slots, element_type *elements,
                                         Initializer initializer) noexcept
        : dimension_{dimension},
          capacity_{capacity},
          slots_{slots},
          elements_{elements},
          initializer_{initializer} {}

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ iterator initial_slot(CG g, key_type const &k, Hash hash) noexcept {
      auto id = (hash(k) + g.thread_rank()) % capacity_;
      return iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ const_iterator initial_slot(CG g, key_type const &k, Hash hash) const noexcept {
      auto id = (hash(k) + g.thread_rank()) % capacity_;
      return const_iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ iterator next_slot(CG g, iterator s) noexcept {
      auto id = (s.index() + g.size()) % capacity_;
      return iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ const_iterator next_slot(CG g, const_iterator s) const noexcept {
      auto id = (s.index() + g.size()) % capacity_;
      return const_iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

   public:
    __device__ size_t get_dimension() const noexcept { return dimension_; }

    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     *
     * @return The maximum number of elements the hash map can hold
     */
    __device__ size_t get_capacity() const noexcept { return capacity_; }

    __device__ Initializer get_initializer() const noexcept { return initializer_; }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    __device__ constexpr key_type get_empty_key_sentinel() const noexcept {
      return empty_key_sentinel;
    }

    __device__ constexpr key_type get_reclaimed_key_sentinel() const noexcept {
      return reclaimed_key_sentinel;
    }

    __device__ iterator at(uint32_t id) noexcept {
      return iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

    __device__ const_iterator at(uint32_t id) const noexcept {
      return const_iterator(slots_ + id, elements_ + dimension_ * id, id);
    }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those familiar with checking
     * an iterator returned from `find()` against the `end()` iterator.
     *
     * @return An iterator to one past the last slot
     */
    __device__ iterator end() noexcept {
      return iterator(slots_ + capacity_, elements_ + dimension_ * capacity_, capacity_);
    }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those
     * familiar with checking an iterator returned from `find()` against the
     * `end()` iterator.
     *
     * @return A const_iterator to one past the last slot
     */
    __device__ const_iterator end() const noexcept {
      return const_iterator(slots_ + capacity_, elements_ + dimension_ * capacity_, capacity_);
    }
  };

 public:
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the map.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.
   */
  class device_mutable_view : public device_view_base {
   public:
    using key_type = typename device_view_base::key_type;
    using element_type = typename device_view_base::element_type;
    using iterator = typename device_view_base::iterator;
    /**
     * @brief Construct a mutable view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     */
    __host__ __device__ device_mutable_view(uint32_t dimension, size_t capacity,
                                            pair_atomic_type *slots, element_type *elements,
                                            Initializer initializer) noexcept
        : device_view_base{dimension, capacity, slots, elements, initializer} {}

   private:
    /**
     * @brief Inserts the specified key/value pair with a CAS of the key and a dependent write of
     * the value.
     *
     * @param current_slot The slot to insert
     * @param insert_pair The pair to insert
     * @return An insert result from the `insert_resullt` enumeration.
     */
    __device__ insert_result try_occupy(iterator current_slot, key_type const &key) noexcept;

   public:
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single insert. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `insert` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @return `true` if the insert was successful, `false` otherwise.
     */
    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ insert_result lookup_or_insert(CG const &g, pair_type const &lookup_or_insert_pair,
                                              Hash hash = Hash{}) noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ insert_result lookup_or_insert(CG const &g,
                                              pointer_pair_type const &lookup_or_insert_pair,
                                              Hash hash = Hash{}) noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool add(CG g, const_pair_type const &add_pair, Hash hash = Hash{}) noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool update(CG g, const_pair_type const &add_pair, Hash hash = Hash{}) noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool try_remove(CG const &g, key_type const &k, Hash hash = Hash{}) noexcept;
  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base {
   public:
    using key_type = typename device_view_base::key_type;
    using element_type = typename device_view_base::element_type;

    /**
     * @brief Construct a view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     */
    __host__ __device__ device_view(uint32_t dimension, size_t capacity, pair_atomic_type *slots,
                                    element_type *elements, Initializer initializer) noexcept
        : device_view_base{dimension, capacity, slots, elements, initializer} {}

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns a const_iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups
     * API to to leverage multiple threads to perform a single find. This
     * provides a significant boost in throughput compared to the non
     * Cooperative Group `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @return An iterator to the position at which the key/value pair containing `k` was inserted
     */
    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool lookup(CG g, pair_type const &lookup_pair, Hash hash = Hash{}) const noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool lookup(CG g, pointer_pair_type const &lookup_pair,
                           Hash hash = Hash{}) const noexcept;

    template <typename CG, typename Hash = cuco::detail::MurmurHash3_32<key_type>>
    __device__ bool contains(CG g, key_type const &k, Hash hash = Hash{}) const noexcept;
  };  // class device_view

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   *
   * @return The number of elements in the map
   */
  size_t get_size() const noexcept { return *h_occupied_size_ - *h_reclaimed_size_; }

  size_t get_occupied_size() const noexcept { return *h_occupied_size_; }

  /**
   * @brief Constructs a device_view object based on the members of the
   * `static_map` object.
   *
   * @return A device_view object based on the members of the `static_map`
   * object
   */
  device_view get_device_view() const noexcept {
    return device_view(dimension_, capacity_, slots_, elements_, initializer_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the
   * `static_map` object
   *
   * @return A device_mutable_view object based on the members of the
   * `static_map` object
   */
  device_mutable_view get_device_mutable_view() const noexcept {
    return device_mutable_view(dimension_, capacity_, slots_, elements_, initializer_);
  }

 private:
  const uint32_t dimension_{};
  const size_t capacity_{};           ///< Total number of slots
  pair_atomic_type *slots_{nullptr};  ///< Pointer to flat slots storage
  element_type *elements_{nullptr};
  atomic_ctr_type *occupied_size_{};
  atomic_ctr_type *reclaimed_size_{};
  atomic_ctr_type *h_occupied_size_{};
  atomic_ctr_type *h_reclaimed_size_{};
  Initializer initializer_{};
};
}  // namespace cuco

#include <cuco/detail/static_map.inl>
