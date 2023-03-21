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
dynamic_map<Key, Element, Initializer>::dynamic_map(uint32_t dimension, size_t initial_capacity,
                                                    Initializer const &initializer)
    : max_load_factor_(0.60),
      min_insert_size_(1E4),
      dimension_(dimension),
      initial_capacity_(initial_capacity),
      capacity_(initial_capacity),
      initializer_(initializer) {}

template <typename Key, typename Element, typename Initializer>
void dynamic_map<Key, Element, Initializer>::initialize(cudaStream_t stream) {
  void *ptr;
  CUCO_CUDA_TRY(cudaMalloc(&ptr, sizeof(atomic_ctr_type) * max_num_submaps * 3));
  occupied_size_per_submap_ = reinterpret_cast<atomic_ctr_type *>(ptr);
  reclaimed_size_per_submap_ = reinterpret_cast<atomic_ctr_type *>(ptr) + max_num_submaps;
  num_successes_ = reinterpret_cast<atomic_ctr_type *>(ptr) + max_num_submaps * 2;

  void *h_ptr;
  CUCO_CUDA_TRY(cudaMallocHost(&h_ptr, sizeof(atomic_ctr_type) * max_num_submaps * 3));
  h_occupied_size_per_submap_ = reinterpret_cast<atomic_ctr_type *>(h_ptr);
  h_reclaimed_size_per_submap_ = reinterpret_cast<atomic_ctr_type *>(h_ptr) + max_num_submaps;
  h_num_successes_ = reinterpret_cast<atomic_ctr_type *>(h_ptr) + max_num_submaps * 2;

  for (size_t i = 0; i < max_num_submaps; ++i) {
    h_occupied_size_per_submap_[i] = 0;
    h_reclaimed_size_per_submap_[i] = 0;
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(ptr, h_ptr, sizeof(atomic_ctr_type) * 2 * max_num_submaps,
                                cudaMemcpyHostToDevice, stream));
}

template <typename Key, typename Element, typename Initializer>
void dynamic_map<Key, Element, Initializer>::uninitialize(cudaStream_t stream) {
  for (auto &submap : submaps_) {
    submap->uninitialize(stream);
  }
  CUCO_ASSERT_CUDA_SUCCESS(cudaFree(reinterpret_cast<void *>(occupied_size_per_submap_)));
  CUCO_ASSERT_CUDA_SUCCESS(cudaFreeHost(reinterpret_cast<void *>(h_occupied_size_per_submap_)));
}

template <typename Key, typename Element, typename Initializer>
void dynamic_map<Key, Element, Initializer>::reserve(size_t n, cudaStream_t stream) {
  int64_t num_elements_remaining = n;
  size_t submap_idx = 0;
  while (num_elements_remaining > 0) {
    size_t submap_capacity;

    // if the submap already exists
    if (submap_idx < submaps_.size()) {
      submap_capacity = submaps_[submap_idx]->get_capacity();
    }
    // if the submap does not exist yet, create it
    else {
      submap_capacity = capacity_;
      auto submap = std::make_unique<static_map<key_type, element_type, Initializer>>(
          dimension_, submap_capacity, occupied_size_per_submap_ + submap_idx,
          reclaimed_size_per_submap_ + submap_idx, h_occupied_size_per_submap_ + submap_idx,
          h_reclaimed_size_per_submap_ + submap_idx, initializer_);
      submap->initialize(stream);
      submap_views_.push_back(submap->get_device_view());
      submap_mutable_views_.push_back(submap->get_device_mutable_view());
      submaps_.push_back(std::move(submap));

      capacity_ *= 2;
    }

    num_elements_remaining -= max_load_factor_ * submap_capacity - min_insert_size_;
    submap_idx++;
  }
}

template <typename Key, typename Element, typename Initializer>
template <typename Hash>
void dynamic_map<Key, Element, Initializer>::lookup(key_type const *keys, element_type *values,
                                                    size_t num_keys, cudaStream_t stream,
                                                    Hash hash) {
  size_t num_to_insert = num_keys;
  reserve(get_size() + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    size_t capacity_remaining =
        max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we
    // can insert only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      auto n = std::min(capacity_remaining, num_to_insert);
      auto const block_size = 128;
      auto const stride = 1;
      auto const tile_size = 4;
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::lookup<block_size, tile_size, pair_type><<<grid_size, block_size, 0, stream>>>(
          keys, values, dimension_, n, submap_views_.data().get(),
          submap_mutable_views_.data().get(), submaps_[submap_idx]->occupied_size_,
          submaps_[submap_idx]->reclaimed_size_, submap_idx, submaps_.size(), hash);

      keys += n;
      values += n * dimension_;
      num_to_insert -= n;
    }
    submap_idx++;
  }

  CUCO_CUDA_TRY(cudaMemcpyAsync(h_occupied_size_per_submap_, occupied_size_per_submap_,
                                sizeof(atomic_ctr_type) * max_num_submaps * 2,
                                cudaMemcpyDeviceToHost, stream));
  // make sure h_occupied_size_per_submap_ and h_reclaimed_size_per_submap_ are valid
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Element, typename Initializer>
template <typename Hash>
void dynamic_map<Key, Element, Initializer>::lookup_unsafe(key_type const *keys,
                                                           element_type **values, size_t num_keys,
                                                           cudaStream_t stream, Hash hash) {
  size_t num_to_insert = num_keys;
  reserve(get_size() + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    size_t capacity_remaining =
        max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we
    // can insert only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      auto n = std::min(capacity_remaining, num_to_insert);
      auto const block_size = 128;
      auto const stride = 1;
      auto const tile_size = 4;
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::lookup_unsafe<block_size, tile_size, pair_type><<<grid_size, block_size, 0, stream>>>(
          keys, values, dimension_, n, submap_views_.data().get(),
          submap_mutable_views_.data().get(), submaps_[submap_idx]->occupied_size_,
          submaps_[submap_idx]->reclaimed_size_, submap_idx, submaps_.size(), hash);

      keys += n;
      values += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }

  CUCO_CUDA_TRY(cudaMemcpyAsync(h_occupied_size_per_submap_, occupied_size_per_submap_,
                                sizeof(atomic_ctr_type) * max_num_submaps * 2,
                                cudaMemcpyDeviceToHost, stream));
  // make sure h_occupied_size_per_submap_ and h_reclaimed_size_per_submap_ are valid
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Element, typename Initializer>
template <typename Hash>
void dynamic_map<Key, Element, Initializer>::scatter_add(key_type const *keys,
                                                         element_type const *updates,
                                                         size_t num_keys, cudaStream_t stream,
                                                         Hash hash) {
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::scatter_add<tile_size, const_pair_type><<<grid_size, block_size, 0, stream>>>(
      keys, updates, dimension_, num_keys, submap_mutable_views_.data().get(), submaps_.size(),
      hash);
}

template <typename Key, typename Element, typename Initializer>
template <typename Hash>
void dynamic_map<Key, Element, Initializer>::scatter_update(key_type const *keys,
                                                            element_type const *updates,
                                                            size_t num_keys, cudaStream_t stream,
                                                            Hash hash) {
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::scatter_update<tile_size, const_pair_type><<<grid_size, block_size, 0, stream>>>(
      keys, updates, dimension_, num_keys, submap_mutable_views_.data().get(), submaps_.size(),
      hash);
}

template <typename Key, typename Element, typename Initializer>
template <typename Hash>
void dynamic_map<Key, Element, Initializer>::remove(key_type const *keys, size_t num_keys,
                                                    cudaStream_t stream, Hash hash) {
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::remove<tile_size>
      <<<grid_size, block_size,
         sizeof(cuda::atomic<size_t, cuda::thread_scope_block>) * submaps_.size(), stream>>>(
          keys, num_keys, submap_mutable_views_.data().get(), reclaimed_size_per_submap_,
          submaps_.size(), hash);

  CUCO_CUDA_TRY(cudaMemcpyAsync(h_reclaimed_size_per_submap_, reclaimed_size_per_submap_,
                                sizeof(atomic_ctr_type) * max_num_submaps, cudaMemcpyDeviceToHost,
                                stream));
  // make sure h_reclaimed_size_per_submap_ is valid
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Element, typename Initializer>
void dynamic_map<Key, Element, Initializer>::eXport(key_type *keys, element_type *values,
                                                    size_t num_keys, cudaStream_t stream) {
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  h_num_successes_[0] = 0;
  CUCO_ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(
      num_successes_, h_num_successes_, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));
  detail::eXport<tile_size><<<grid_size, block_size, 0, stream>>>(
      keys, values, num_keys, num_successes_, submap_views_.data().get(), submaps_.size());

  CUCO_CUDA_TRY(cudaMemcpyAsync(h_num_successes_, num_successes_,
                                sizeof(atomic_ctr_type) * max_num_submaps, cudaMemcpyDeviceToHost,
                                stream));
  // make sure h_num_successes_ is valid
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key, typename Element, typename Initializer>
void dynamic_map<Key, Element, Initializer>::clear(cudaStream_t stream) {
  auto const num_keys = get_size();
  auto const block_size = 256;
  auto const stride = 1;
  auto const grid_size = (num_keys + stride * block_size - 1) / (stride * block_size);

  detail::clear<block_size, Key, Element><<<grid_size, block_size, 0, stream>>>(
      submap_mutable_views_.data().get(), submap_mutable_views_.size());
  for (size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    submaps_[submap_idx]->occupied_size_ = 0;
    submaps_[submap_idx]->reclaimed_size_ = 0;

    occupied_size_per_submap_[submap_idx] = 0;
    reclaimed_size_per_submap_[submap_idx] = 0;
  }
}

}  // namespace cuco
