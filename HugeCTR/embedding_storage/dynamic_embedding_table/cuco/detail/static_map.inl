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

template <typename Key, typename Element, typename Initializer>
static_map<Key, Element, Initializer>::static_map(uint32_t dimension, size_t capacity, atomic_ctr_type *occupied_size, atomic_ctr_type *reclaimed_size, atomic_ctr_type *h_occupied_size, atomic_ctr_type *h_reclaimed_size, Initializer initializer)
    : dimension_(dimension), capacity_(capacity), occupied_size_(occupied_size), reclaimed_size_(reclaimed_size), h_occupied_size_(h_occupied_size), h_reclaimed_size_(h_reclaimed_size), initializer_(initializer) {}

template <typename Key, typename Element, typename Initializer>
void static_map<Key, Element, Initializer>::initialize(cudaStream_t stream) {
  printf("static_map allocated, size=%zu\n", (sizeof(pair_atomic_type) + sizeof(element_type) * dimension_) * capacity_);
  CUCO_CUDA_TRY(cudaMallocAsync(&slots_, sizeof(pair_atomic_type) * capacity_, stream));
  CUCO_CUDA_TRY(cudaMallocAsync(&elements_, sizeof(element_type) * dimension_ * capacity_, stream));
  auto constexpr block_size = 1024;
  auto const grid_size = (capacity_ + block_size - 1) / block_size;
  detail::initialize<atomic_key_type, element_type, lock_type><<<grid_size, block_size, 0, stream>>>(slots_, elements_, empty_key_sentinel, dimension_, capacity_);
}

template <typename Key, typename Element, typename Initializer>
void static_map<Key, Element, Initializer>::uninitialize(cudaStream_t stream) {
  cudaFreeAsync(slots_, stream);
  cudaFreeAsync(elements_, stream);
}

template <typename Key, typename Element, typename Initializer>
__device__ insert_result static_map<Key, Element, Initializer>::device_mutable_view::try_occupy(iterator current_slot, key_type const &key) noexcept {

  auto expected_key = this->get_empty_key_sentinel();
  if (current_slot.key().compare_exchange_strong(expected_key, key, cuda::std::memory_order_relaxed)) {
    return insert_result::OCCUPIED_EMPTY;
  }

  if (expected_key == this->get_reclaimed_key_sentinel()) {
    if (current_slot.key().compare_exchange_strong(expected_key, key, cuda::std::memory_order_relaxed)) {
      return insert_result::OCCUPIED_RECLAIMED;
    }
  }

  // our key was already present in the slot, so our key is a duplicate
  if (expected_key == key) {
    return insert_result::DUPLICATE;
  }

  return insert_result::CONTINUE;
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ insert_result static_map<Key, Element, Initializer>::device_mutable_view::lookup_or_insert(CG const &g, pair_type const &lookup_or_insert_pair, Hash hash) noexcept {
  auto current_slot = initial_slot(g, lookup_or_insert_pair.first, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    auto const window_contains_exist = g.ballot(existing_key == lookup_or_insert_pair.first);

    if (window_contains_exist) {
      auto const src_lane = __ffs(window_contains_exist) - 1;
      auto const value = g.shfl(current_slot.value(), src_lane);
      current_slot.lock().acquire(g, src_lane);
      detail::copy_array(g, this->get_dimension(), lookup_or_insert_pair.second, value);
      current_slot.lock().release(g, src_lane);
      return insert_result::DUPLICATE;
    }

    auto const window_contains_empty = g.ballot(existing_key == this->get_empty_key_sentinel() || existing_key == this->get_reclaimed_key_sentinel());

    // we found an empty slot, but not the key we are inserting, so this must be an empty slot into which we can insert the key
    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      auto const src_lane = __ffs(window_contains_empty) - 1;

      if (g.thread_rank() == src_lane) {
        status = try_occupy(current_slot, lookup_or_insert_pair.first);
      }

      status = g.shfl(status, src_lane);

      if (status == insert_result::OCCUPIED_EMPTY || status == insert_result::OCCUPIED_RECLAIMED) {
        auto const value = g.shfl(current_slot.value(), src_lane);
        detail::init_and_copy_array(g, this->get_dimension(), value, lookup_or_insert_pair.second, this->get_initializer());
        current_slot.lock().release(g, src_lane);
      } else if (status == insert_result::DUPLICATE) {
        auto const value = g.shfl(current_slot.value(), src_lane);
        current_slot.lock().acquire(g, src_lane);
        detail::copy_array(g, this->get_dimension(), lookup_or_insert_pair.second, value);
        current_slot.lock().release(g, src_lane);
      }

      // successful insert
      if (status != insert_result::CONTINUE) {
        return status;
      }
      // if we've gotten this far, a different key took our spot before we could insert. We need to retry the insert on the same window
    }
    // if there are no empty slots in the current window, we move onto the next window
    else {
      current_slot = next_slot(g, current_slot);
    }
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ auto static_map<Key, Element, Initializer>::device_mutable_view::lookup_or_insert_unsafe(CG const &g, key_type const &lookup_or_insert_key, Hash hash) noexcept -> std::pair<element_type*, insert_result> {
  auto current_slot = initial_slot(g, lookup_or_insert_key, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    auto const window_contains_exist = g.ballot(existing_key == lookup_or_insert_key);

    if (window_contains_exist) {
      auto const src_lane = __ffs(window_contains_exist) - 1;
      auto const value = g.shfl(current_slot.value(), src_lane);
      return {value, insert_result::DUPLICATE};
    }

    auto const window_contains_empty = g.ballot(existing_key == this->get_empty_key_sentinel() || existing_key == this->get_reclaimed_key_sentinel());

    // we found an empty slot, but not the key we are inserting, so this must be an empty slot into which we can insert the key
    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      auto const src_lane = __ffs(window_contains_empty) - 1;

      if (g.thread_rank() == src_lane) {
        status = try_occupy(current_slot, lookup_or_insert_key);
      }

      status = g.shfl(status, src_lane);

      element_type* insert_ptr = nullptr;
      if (status == insert_result::OCCUPIED_EMPTY || status == insert_result::OCCUPIED_RECLAIMED) {
        auto const value = g.shfl(current_slot.value(), src_lane);
        detail::init_array(g, this->get_dimension(), value, this->get_initializer());
        current_slot.lock().release(g, src_lane);
        insert_ptr = value;
      } else if (status == insert_result::DUPLICATE) {
        auto const value = g.shfl(current_slot.value(), src_lane);
        insert_ptr = value;
      }

      // successful insert
      if (status != insert_result::CONTINUE) {
        assert(insert_ptr != nullptr);
        return {insert_ptr, status};
      }
      // if we've gotten this far, a different key took our spot before we could insert. We need to retry the insert on the same window
    }
    // if there are no empty slots in the current window, we move onto the next window
    else {
      current_slot = next_slot(g, current_slot);
    }
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ bool static_map<Key, Element, Initializer>::device_mutable_view::add(CG g, const_pair_type const &add_pair, Hash hash) noexcept {
  auto current_slot = initial_slot(g, add_pair.first, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    // the key we were searching for was found by one of the threads, so we return an iterator to the entry
    auto const exists = g.ballot(existing_key == add_pair.first);
    if (exists) {
      auto const src_lane = __ffs(exists) - 1;
      auto const value = g.shfl(current_slot.value(), src_lane);
      current_slot.lock().acquire(g, src_lane);
      detail::accumulate_array(g, this->get_dimension(), value, add_pair.second);
      current_slot.lock().release(g, src_lane);
      return true;
    }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.any(existing_key == this->get_empty_key_sentinel())) {
      return false;
    }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next window
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ bool static_map<Key, Element, Initializer>::device_view::lookup(CG g, pair_type const &lookup_pair, Hash hash) const noexcept {
  auto current_slot = initial_slot(g, lookup_pair.first, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    // the key we were searching for was found by one of the threads, so we return an iterator to the entry
    auto const exists = g.ballot(existing_key == lookup_pair.first);
    if (exists) {
      auto const src_lane = __ffs(exists) - 1;
      auto const value = g.shfl(current_slot.value(), src_lane);
      current_slot.lock().acquire(g, src_lane);
      detail::copy_array(g, this->get_dimension(), lookup_pair.second, value);
      current_slot.lock().release(g, src_lane);
      return true;
    }

    // we found an empty slot, meaning that the key we're searching for isn't in this submap, so we should move onto the next one
    if (g.any(existing_key == this->get_empty_key_sentinel())) {
      return false;
    }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next window in the current submap
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ auto static_map<Key, Element, Initializer>::device_view::lookup_unsafe(CG g, key_type const &lookup_key, Hash hash) const noexcept -> element_type* {
  auto current_slot = initial_slot(g, lookup_key, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    auto const exists = g.ballot(existing_key == lookup_key);
    if (exists) {
      auto const src_lane = __ffs(exists) - 1;
      element_type *value = const_cast<element_type *>(g.shfl(current_slot.value(), src_lane));
      return value;
    }

    // we found an empty slot, meaning that the key we're searching for isn't in this submap, so we
    // should move onto the next one
    if (g.any(existing_key == this->get_empty_key_sentinel())) {
      return nullptr;
    }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window in the current submap
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG, typename Hash>
__device__ bool static_map<Key, Element, Initializer>::device_mutable_view::try_remove(
    CG const &g, key_type const &key, Hash hash) noexcept {
  auto current_slot = initial_slot(g, key, hash);

  while (true) {
    auto const existing_key = current_slot.key().load(cuda::std::memory_order_relaxed);

    auto const window_matches = g.ballot(existing_key == key);

    // we found an empty slot, but not the key we are inserting, so this must be an empty slot into
    // which we can insert the key
    if (window_matches) {
      // the first lane in the group with an empty slot will attempt the insert
      auto const src_lane = __ffs(window_matches) - 1;

      if (g.thread_rank() == src_lane) {
        current_slot.key().store(this->get_reclaimed_key_sentinel(),
                                 cuda::std::memory_order_relaxed);
      }

      return true;
    }

    if (g.any(existing_key == this->get_empty_key_sentinel())) {
      return false;
    }

    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename CG>
__device__ bool static_map<Key, Element, Initializer>::device_mutable_view::clear(CG const &g) noexcept {
  const auto k = this->get_empty_key_sentinel();
  const auto tid = g.thread_rank();
  
  for (int i = tid; i < this->get_capacity(); i += g.size()) {
    iterator slot = this->at(i);
    slot.key().store(k, cuda::std::memory_order_relaxed);
    i += g.size();
  }
}

} // namespace cuco
