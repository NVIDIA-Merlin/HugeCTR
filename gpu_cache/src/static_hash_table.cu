/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>
#include <cuda.h>
#include <linux/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <static_hash_table.hpp>

namespace gpu_cache {

template <typename T>
__device__ __forceinline__ T atomicCASHelper(T *address, T compare, T val) {
  return atomicCAS(address, compare, val);
}

template <>
__device__ __forceinline__ long long atomicCASHelper(long long *address, long long compare,
                                                     long long val) {
  return (long long)atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

template <>
__device__ __forceinline__ int64_t atomicCASHelper(int64_t *address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                            (unsigned long long)val);
}

template <unsigned int group_size, typename key_type, typename size_type, typename hasher,
          typename CG>
__device__ size_type insert(key_type *table, int64_t capacity, key_type key, const hasher &hash,
                            const CG &cg, const key_type empty_key, const size_type invalid_slot) {
  // If insert successfully, return its position in the table,
  // otherwise return invalid_slot.

  const size_type num_groups = capacity / group_size;
#if (CUDA_VERSION < 11060)
  unsigned long long num_threads_per_group = cg.size();
#else
  unsigned long long num_threads_per_group = cg.num_threads();
#endif
  const unsigned int num_tiles_per_group = group_size / num_threads_per_group;

  // Assuming capacity is a power of 2
  size_type slot = hash(key) & (capacity - 1);
  slot = slot - (slot & (size_type)(group_size - 1)) + cg.thread_rank();

  for (size_t step = 0; step < static_cast<int64_t>(num_groups); ++step) {
    for (unsigned int i = 0; i < num_tiles_per_group; ++i) {
      key_type existed_key = table[slot];

      // Check if key already exists
      bool existed = cg.any(existed_key == key);
      if (existed) {
        return invalid_slot;
      }

      // Try to insert the target key into empty slot
      while (true) {
        int can_insert = cg.ballot(existed_key == empty_key);

        if (!can_insert) {
          break;
        }

        bool succeed = false;
        int src_lane = __ffs(can_insert) - 1;

        if (cg.thread_rank() == src_lane) {
          key_type old = atomicCASHelper(table + slot, empty_key, key);
          if (old == empty_key) {
            // Insert key successfully
            succeed = true;
          } else if (old == key) {
            // The target key was inserted by another thread
            succeed = true;
            slot = invalid_slot;
          } else {
            // The empty slot was occupied by another key,
            // update the existed_key for next loop.
            existed_key = old;
          }
        }

        succeed = cg.shfl(succeed, src_lane);
        if (succeed) {
          slot = cg.shfl(slot, src_lane);
          return slot;
        }
      }

      slot += num_threads_per_group;
    }
    slot = (slot + group_size * step) & (capacity - 1);
  }

  return invalid_slot;
}

template <unsigned int tile_size, unsigned int group_size, typename key_type, typename size_type,
          typename hasher>
__global__ void InsertKeyKernel(key_type *table_keys, size_type *table_indices, int64_t capacity,
                                const key_type *keys, size_type num_keys, int64_t offset,
                                hasher hash, const key_type empty_key,
                                const size_type invalid_slot) {
  static_assert(tile_size <= group_size, "tile_size cannot be larger than group_size");

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  int tile_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
  int tile_cnt = tile.meta_group_size() * gridDim.x;

  for (size_t i = tile_idx; i < static_cast<size_t>(num_keys); i += tile_cnt) {
    key_type key = keys[i];
    if (key == empty_key) {
      if (tile.thread_rank() == 0 && table_keys[capacity] != empty_key) {
        table_keys[capacity] = empty_key;
        table_indices[capacity] = i + offset;
      }
      continue;
    }
    size_type slot =
        insert<group_size>(table_keys, capacity, key, hash, tile, empty_key, invalid_slot);
    if (tile.thread_rank() == 0 && slot != invalid_slot) {
      table_indices[slot] = i + offset;
    }
  }
}

template <unsigned int group_size, typename key_type, typename size_type, typename hasher,
          typename CG>
__device__ size_type lookup(key_type *table, int64_t capacity, key_type key, const hasher &hash,
                            const CG &cg, const key_type empty_key, const size_type invalid_slot) {
  // If lookup successfully, return the target key's position in the table,
  // otherwise return invalid_slot.

  const size_type num_groups = capacity / group_size;

#if (CUDA_VERSION < 11060)
  unsigned long long num_threads_per_group = cg.size();
#else
  unsigned long long num_threads_per_group = cg.num_threads();
#endif

  const unsigned int num_tiles_per_group = group_size / num_threads_per_group;

  // Assuming capacity is a power of 2
  size_type slot = hash(key) & (capacity - 1);
  slot = slot - (slot & (size_type)(group_size - 1)) + cg.thread_rank();

  for (size_type step = 0; step < num_groups; ++step) {
    for (unsigned int i = 0; i < num_tiles_per_group; ++i) {
      key_type existed_key = table[slot];

      // Check if key exists
      int existed = cg.ballot(existed_key == key);
      if (existed) {
        int src_lane = __ffs(existed) - 1;
        slot = cg.shfl(slot, src_lane);
        return slot;
      }

      // The target key doesn't exist
      bool contain_empty = cg.any(existed_key == empty_key);
      if (contain_empty) {
        return invalid_slot;
      }

      slot += num_threads_per_group;
    }
    slot = (slot + group_size * step) & (capacity - 1);
  }

  return invalid_slot;
}

template <int warp_size, typename value_type, typename out_value_type>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               out_value_type *d_dst, const value_type *d_src) {
  // 16 bytes align
  if (emb_vec_size_in_float % 4 != 0 || (size_t)d_dst % 16 != 0 || (size_t)d_src % 16 != 0) {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
      d_dst[i] = out_value_type(d_src[i]);
    }
  } else {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float / 4; i += warp_size) {
      *(float4 *)(d_dst + i * 4) = __ldg((const float4 *)(d_src + i * 4));
    }
  }
}

template <int warp_size, typename value_type, typename out_value_type>
__forceinline__ __device__ void warp_tile_quant_copy(const size_t lane_idx,
                                                     const size_t emb_vec_size_in_float,
                                                     out_value_type *d_dst, const value_type *d_src,
                                                     const float scale) {
  // Todo:Vectorized read of char4
  if (emb_vec_size_in_float % 4 != 0 || (size_t)d_dst % 16 != 0 || (size_t)d_src % 16 != 0) {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
      d_dst[i] = out_value_type(float(__nv_fp8_e4m3(d_src[i])) * scale);
    }
  } else {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float / 4; i += warp_size) {
      char4 tmp = __ldg((const char4 *)(d_src + i * 4));
      float4 float4Tmp;
      float4Tmp.x = ((float)reinterpret_cast<const __nv_fp8_e4m3 *>(&tmp.x)[0]) * scale;
      float4Tmp.y = ((float)reinterpret_cast<const __nv_fp8_e4m3 *>(&tmp.y)[0]) * scale;
      float4Tmp.z = ((float)reinterpret_cast<const __nv_fp8_e4m3 *>(&tmp.z)[0]) * scale;
      float4Tmp.w = ((float)reinterpret_cast<const __nv_fp8_e4m3 *>(&tmp.w)[0]) * scale;
      *(float4 *)(d_dst + i * 4) = float4Tmp;
    }
  }
}

template <int warp_size, typename value_type, typename out_value_type>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               out_value_type *d_dst,
                                               out_value_type default_value) {
#pragma unroll
  for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
    d_dst[i] = out_value_type(default_value);
  }
}

template <unsigned int tile_size, unsigned int group_size, typename key_type, typename value_type,
          typename out_value_type, typename size_type, typename hasher>
__global__ void LookupKernel(key_type *table_keys, size_type *table_indices, float *quant_scales_,
                             int64_t capacity, const key_type *keys, int num_keys,
                             const value_type *values, int value_dim, out_value_type *output,
                             hasher hash, const key_type empty_key, out_value_type default_value,
                             const size_type invalid_slot) {
  static_assert(tile_size <= group_size, "tile_size cannot be larger than group_size");
  constexpr int WARP_SIZE = 32;
  static_assert(WARP_SIZE % tile_size == 0, "tile_size must be divisible by warp_size");

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);
  auto warp_tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

  int tile_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
  int tile_cnt = tile.meta_group_size() * gridDim.x;

  for (int it = 0; it < (num_keys - 1) / tile_cnt + 1; it++) {
    size_type slot = invalid_slot;
    int key_num = it * tile_cnt + tile_idx;
    if (key_num < num_keys) {
      key_type key = keys[key_num];
      if (key == empty_key) {
        if (tile.thread_rank() == 0 && table_keys[capacity] == key) {
          slot = capacity;
        }
      } else {
        slot = lookup<group_size>(table_keys, capacity, key, hash, tile, empty_key, invalid_slot);
      }
    }
    for (int i = 0; i < WARP_SIZE / tile_size; i++) {
      auto slot_to_read = warp_tile.shfl(slot, i * tile_size);
      int idx_to_write = warp_tile.shfl(key_num, 0) + i;
      if (idx_to_write >= num_keys) break;
      if (slot_to_read == invalid_slot) {
        warp_tile_copy<WARP_SIZE, value_type>(warp_tile.thread_rank(), value_dim,
                                              output + (size_t)value_dim * idx_to_write,
                                              default_value);
        continue;
      }
      auto index = table_indices[slot_to_read];

      if constexpr (nv::is_fp8<value_type>::value) {
        float scale;
        scale = quant_scales_[index];
        warp_tile_quant_copy<WARP_SIZE, value_type>(warp_tile.thread_rank(), value_dim,
                                                    output + (size_t)value_dim * idx_to_write,
                                                    values + (size_t)value_dim * index, scale);
      } else {
        warp_tile_copy<WARP_SIZE, value_type>(warp_tile.thread_rank(), value_dim,
                                              output + (size_t)value_dim * idx_to_write,
                                              values + (size_t)value_dim * index);
      }
    }
  }
}

template <typename key_type, typename value_type, typename out_value_type, unsigned int tile_size,
          unsigned int group_size, typename hasher>
StaticHashTable<key_type, value_type, out_value_type, tile_size, group_size,
                hasher>::StaticHashTable(size_type capacity, int value_dim, bool enable_pagelock,
                                         hasher hash)
    : table_keys_(nullptr),
      table_indices_(nullptr),
      key_capacity_(capacity * 2),
      table_values_(nullptr),
      value_capacity_(capacity),
      value_dim_(value_dim),
      size_(0),
      enable_pagelock(enable_pagelock),
      hash_(hash) {
  // Check parameters
  if (capacity <= 0) {
    printf("Error: capacity must be larger than 0\n");
    exit(EXIT_FAILURE);
  }
  if (value_dim <= 0) {
    printf("Error: value_dim must be larger than 0\n");
    exit(EXIT_FAILURE);
  }

  // Make key_capacity_ be a power of 2
  int64_t new_capacity = group_size;
  while (new_capacity < key_capacity_) {
    new_capacity *= 2;
  }
  key_capacity_ = new_capacity;

  // Allocate device memory
  size_t align_m = 16;
  size_t num_keys = key_capacity_ + 1;
  size_t num_scales = key_capacity_ + 1;
  size_t num_values = (value_capacity_ * value_dim_ + align_m - 1) / align_m * align_m;
  CUDA_CHECK(cudaMalloc(&table_keys_, sizeof(key_type) * num_keys));
  CUDA_CHECK(cudaMalloc(&table_indices_, sizeof(size_type) * num_keys));
// Allocate embedding on host
#ifdef USE_HUGE_PAGES
  // Align size at hugepage boundaries.
  constexpr size_t page_size_mb{2};
  size_t n = sizeof(value_type) * num_values;
  int page_shift{20};
  for (size_t ps{page_size_mb}; (ps & 1) == 0; ps >>= 1) {
    ++page_shift;
  }
  constexpr size_t page_size{page_size_mb * 1024 * 1024};
  const size_t num_pages{(n + page_size - 1) / page_size};
  n = num_pages * page_size;

  // Fetch memory.
  if (n != 0) {
    void *const p{mmap(nullptr, n, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_UNINITIALIZED |
                           (page_shift << HUGETLB_FLAG_ENCODE_SHIFT),
                       -1, 0)};
    if (p == MAP_FAILED) {
      printf("Error: mmap allocation failed! (error = %s)\n", std::strerror(errno));
      std::abort();
    }
    CUDA_CHECK(cudaHostRegister(p, n, cudaHostRegisterMapped));
    table_values_ = static_cast<value_type *>(p);
  }
#else
  if (enable_pagelock) {
    CUDA_CHECK(
        cudaHostAlloc(&table_values_, sizeof(value_type) * num_values, cudaHostAllocPortable));
  } else {
    CUDA_CHECK(cudaMalloc(&table_values_, sizeof(value_type) * num_values));
  }
  if constexpr (nv::is_fp8<value_type>::value) {
    CUDA_CHECK(cudaMalloc(&quant_scales_, sizeof(float) * num_keys));
  }
#endif

  // Initialize table_keys_
  CUDA_CHECK(cudaMemset(table_keys_, 0xff, sizeof(key_type) * key_capacity_));
  CUDA_CHECK(cudaMemset(table_keys_ + key_capacity_, 0, sizeof(key_type)));
}

template <typename key_type, typename value_type, typename out_value_type, unsigned int tile_size,
          unsigned int group_size, typename hasher>
void StaticHashTable<key_type, value_type, out_value_type, tile_size, group_size, hasher>::insert(
    const key_type *keys, const value_type *values, size_type num_keys, cudaStream_t stream,
    const float *quant_scales) {
  if (num_keys == 0) {
    return;
  }
  if (num_keys <= 0 || (size() + num_keys) > capacity()) {
    printf("Error: Invalid num_keys to insert\n");
    exit(EXIT_FAILURE);
  }

  // Insert keys
  constexpr int block = 256;
  int grid = (num_keys - 1) / block + 1;
  InsertKeyKernel<tile_size, group_size>
      <<<grid, block, 0, stream>>>(table_keys_, table_indices_, key_capacity_, keys, num_keys,
                                   size_, hash_, empty_key, invalid_slot);

  // Copy values
  CUDA_CHECK(cudaMemcpyAsync(table_values_ + size_ * value_dim_, values,
                             sizeof(value_type) * num_keys * value_dim_, cudaMemcpyDefault,
                             stream));

  if constexpr (nv::is_fp8<value_type>::value) {
    CUDA_CHECK(cudaMemcpyAsync(quant_scales_ + size_, quant_scales, sizeof(float) * num_keys,
                               cudaMemcpyDefault, stream));
  }
  size_ += num_keys;
}

template <typename key_type, typename value_type, typename out_value_type, unsigned int tile_size,
          unsigned int group_size, typename hasher>
void StaticHashTable<key_type, value_type, out_value_type, tile_size, group_size, hasher>::clear(
    cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(table_keys_, 0xff, sizeof(key_type) * key_capacity_, stream));
  CUDA_CHECK(cudaMemsetAsync(table_keys_ + key_capacity_, 0, sizeof(key_type), stream));
  size_ = 0;
}

template <typename key_type, typename value_type, typename out_value_type, unsigned int tile_size,
          unsigned int group_size, typename hasher>
StaticHashTable<key_type, value_type, out_value_type, tile_size, group_size,
                hasher>::~StaticHashTable() {
  CUDA_CHECK(cudaFree(table_keys_));
  CUDA_CHECK(cudaFree(table_indices_));

#ifdef USE_HUGE_PAGES
  if (table_values_) {
    CUDA_CHECK(cudaHostUnregister(table_values_));
    size_t align_m = 16;
    size_t num_values = (value_capacity_ * value_dim_ + align_m - 1) / align_m * align_m;
    size_t n = sizeof(value_type) * num_values;
    constexpr size_t page_size_mb{2};
    constexpr size_t page_size{page_size_mb * 1024 * 1024};
    const size_t num_pages{(n + page_size - 1) / page_size};
    n = num_pages * page_size;
    if (munmap(table_values_, n) != 0) {
      printf("Error: mmap free failed! (error = %s)\n", std::strerror(errno));
      std::abort();
    }
    table_values_ = nullptr;
  }
#else
  if constexpr (nv::is_fp8<value_type>::value) {
    CUDA_CHECK(cudaFree(quant_scales_));
  } else {
    if (enable_pagelock)
      CUDA_CHECK(cudaFreeHost(table_values_))
    else
      CUDA_CHECK(cudaFree(table_values_));
  }
#endif
}

template <typename key_type, typename value_type, typename out_value_type, unsigned int tile_size,
          unsigned int group_size, typename hasher>
void StaticHashTable<key_type, value_type, out_value_type, tile_size, group_size, hasher>::lookup(
    const key_type *keys, out_value_type *values, int num_keys, out_value_type default_value,
    cudaStream_t stream) {
  if (num_keys == 0) {
    return;
  }

  constexpr int block = 256;
  const int grid = (num_keys - 1) / block + 1;
  // Lookup keys
  LookupKernel<tile_size, group_size><<<grid, block, 0, stream>>>(
      table_keys_, table_indices_, quant_scales_, key_capacity_, keys, num_keys, table_values_,
      value_dim_, values, hash_, empty_key, default_value, invalid_slot);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template class StaticHashTable<uint32_t, float, float>;
template class StaticHashTable<uint32_t, __nv_fp8_e4m3, float>;
template class StaticHashTable<uint32_t, __nv_fp8_e4m3, __nv_fp8_e4m3>;
template class StaticHashTable<long long, float, float>;
template class StaticHashTable<long long, __nv_fp8_e4m3, float>;
template class StaticHashTable<long long, __nv_fp8_e4m3, __nv_fp8_e4m3>;
}  // namespace gpu_cache
