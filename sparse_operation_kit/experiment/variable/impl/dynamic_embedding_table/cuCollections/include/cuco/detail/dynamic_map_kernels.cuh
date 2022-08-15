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

// clang-format off
#include <cooperative_groups.h>
#include <cuda/std/atomic>
// clang-format on

#include <cuco/detail/insert_result.cuh>
#include <cuco/detail/utils.cuh>

namespace cuco {
namespace detail {

#pragma nv_diag_suppress static_var_with_dynamic_init

template <uint32_t block_size, uint32_t tile_size, typename pair_type, typename key_type,
          typename element_type, typename viewT, typename mutableViewT, typename atomicT,
          typename Hash>
__global__ void lookup(key_type const *keys, element_type *values, uint32_t dimension,
                       size_t num_keys, viewT *submap_views, mutableViewT *submap_mutable_views,
                       atomicT *occupied_size_per_submap_, atomicT *reclaimed_size_per_submap_,
                       uint32_t insert_idx, uint32_t num_submaps, Hash hash) {
  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_num_occupied_empty;
  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_num_occupied_reclaimed;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  if (block.thread_rank() == 0) {
    block_num_occupied_empty = 0;
    block_num_occupied_reclaimed = 0;
  }

  block.sync();

  for (auto key_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
       key_idx < num_keys; key_idx += tile.meta_group_size() * grid.group_dim().x) {
    pair_type lookup_pair = {keys[key_idx], values + dimension * key_idx};
    auto exists = false;

    // manually check for duplicates in those submaps we are not inserting into
    for (auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      if (submap_view.lookup(tile, lookup_pair, hash)) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      auto const status =
          submap_mutable_views[insert_idx].lookup_or_insert(tile, lookup_pair, hash);
      if (tile.thread_rank() == 0) {
        if (status == insert_result::OCCUPIED_EMPTY) {
          block_num_occupied_empty++;
        } else if (status == insert_result::OCCUPIED_RECLAIMED) {
          block_num_occupied_reclaimed++;
        }
      }
    }
  }

  block.sync();

  if (block.thread_rank() == 0) {
    *occupied_size_per_submap_ += block_num_occupied_empty;
    *reclaimed_size_per_submap_ -= block_num_occupied_reclaimed;
  }
}

template <uint32_t block_size, uint32_t tile_size, typename pair_type, typename key_type,
          typename element_type, typename viewT, typename mutableViewT, typename atomicT,
          typename Hash>
__global__ void lookup(key_type const *keys, element_type **values, uint32_t dimension,
                       size_t num_keys, viewT *submap_views, mutableViewT *submap_mutable_views,
                       atomicT *occupied_size_per_submap_, atomicT *reclaimed_size_per_submap_,
                       uint32_t insert_idx, uint32_t num_submaps, Hash hash) {
  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_num_occupied_empty;
  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_num_occupied_reclaimed;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  if (block.thread_rank() == 0) {
    block_num_occupied_empty = 0;
    block_num_occupied_reclaimed = 0;
  }

  block.sync();

  for (auto key_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
       key_idx < num_keys; key_idx += tile.meta_group_size() * grid.group_dim().x) {
    pair_type lookup_pair = {keys[key_idx], values + key_idx};
    auto exists = false;

    // manually check for duplicates in those submaps we are not inserting into
    for (auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      if (submap_view.lookup(tile, lookup_pair, hash)) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      auto const status =
          submap_mutable_views[insert_idx].lookup_or_insert(tile, lookup_pair, hash);
      if (tile.thread_rank() == 0) {
        if (status == insert_result::OCCUPIED_EMPTY) {
          block_num_occupied_empty++;
        } else if (status == insert_result::OCCUPIED_RECLAIMED) {
          block_num_occupied_reclaimed++;
        }
      }
    }
  }

  block.sync();

  if (block.thread_rank() == 0) {
    *occupied_size_per_submap_ += block_num_occupied_empty;
    *reclaimed_size_per_submap_ -= block_num_occupied_reclaimed;
  }
}

template <uint32_t tile_size, typename pair_type, typename key_type, typename element_type,
          typename mutableViewT, typename Hash>
__global__ void scatter_add(key_type const *keys, element_type const *updates, uint32_t dimension,
                            size_t num_keys, mutableViewT *submap_mutable_views,
                            uint32_t num_submaps, Hash hash) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  for (auto key_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
       key_idx < num_keys; key_idx += tile.meta_group_size() * grid.group_dim().x) {
    pair_type add_pair = {keys[key_idx], updates + dimension * key_idx};

    for (auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_mutable_views[i];
      if (submap_view.add(tile, add_pair, hash)) {
        break;
      }
    }
  }
}

template <uint32_t tile_size, typename pair_type, typename key_type, typename element_type,
          typename mutableViewT, typename Hash>
__global__ void scatter_update(key_type const *keys, element_type const *updates,
                               uint32_t dimension, size_t num_keys,
                               mutableViewT *submap_mutable_views, uint32_t num_submaps,
                               Hash hash) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  for (auto key_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
       key_idx < num_keys; key_idx += tile.meta_group_size() * grid.group_dim().x) {
    pair_type add_pair = {keys[key_idx], updates + dimension * key_idx};

    for (auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_mutable_views[i];
      if (submap_view.update(tile, add_pair, hash)) {
        break;
      }
    }
  }
}

template <uint32_t tile_size, typename key_type, typename mutableViewT, typename atomicT,
          typename Hash>
__global__ void remove(key_type const *keys, size_t num_keys, mutableViewT *submap_mutable_views,
                       atomicT *reclaimed_size_per_submap_, uint32_t num_submaps, Hash hash) {
  extern __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_num_occupied_reclaimed[];

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  if (block.thread_rank() < num_submaps) block_num_occupied_reclaimed[block.thread_rank()] = 0;

  block.sync();

  for (auto key_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
       key_idx < num_keys; key_idx += tile.meta_group_size() * grid.group_dim().x) {
    key_type key = keys[key_idx];

    // manually check for duplicates in those submaps we are not inserting into
    for (auto i = 0; i < num_submaps; ++i) {
      if (submap_mutable_views[i].try_remove(tile, key, hash)) {
        if (tile.thread_rank() == 0) block_num_occupied_reclaimed[i] += 1;
        break;
      }
    }
  }

  block.sync();

  if (block.thread_rank() < num_submaps)
    reclaimed_size_per_submap_[block.thread_rank()] +=
        block_num_occupied_reclaimed[block.thread_rank()];
}

template <uint32_t tile_size, typename key_type, typename value_type, typename viewT,
          typename atomicT>
__global__ void eXport(key_type *keys, value_type *values, size_t num_keys, atomicT *counter,
                       viewT *submap_views, uint32_t num_submaps) {
  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> block_counter;
  __shared__ size_t global_offset;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  for (auto i = 0; i < num_submaps; ++i) {
    auto const submap_view = submap_views[i];

    for (auto j = grid.thread_rank(); j < submap_view.get_capacity(); j += grid.size()) {
      if (block.thread_rank() == 0) {
        block_counter = 0;
      }

      block.sync();

      auto it = submap_view.at(j);

      int block_offset = -1;

      // Place any valid entry into the next avaiable slot in shared memory
      if (it.key() != submap_view.get_empty_key_sentinel() &&
          it.key() != submap_view.get_reclaimed_key_sentinel()) {
        block_offset = block_counter.fetch_add(1, cuda::std::memory_order_relaxed);
      }

      block.sync();

      // Write back coalesced entries from shared memory to global memory
      if (block.thread_rank() == 0) {
        auto block_num_entries = block_counter.load(cuda::std::memory_order_relaxed);
        global_offset = counter[0].fetch_add(block_num_entries, cuda::std::memory_order_relaxed);
      }

      block.sync();

      if (block_offset != -1) {
        auto offset = global_offset + block_offset;
        if (offset < num_keys) {
          keys[offset] = it.key();

          detail::copy_array(cooperative_groups::this_thread(), submap_view.get_dimension(),
                             values + submap_view.get_dimension() * offset, it.value());
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace cuco