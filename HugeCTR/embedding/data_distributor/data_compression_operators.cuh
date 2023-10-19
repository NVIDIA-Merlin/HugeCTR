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
#pragma once

#include <nccl.h>

#include <core/core.hpp>
#include <core23/cuda_primitives.cuh>
#include <embedding/common.hpp>
#include <embedding/data_distributor/data_compression_operators.hpp>
#include <gpu_cache/include/hash_functions.cuh>
#include <memory>
#include <utils.cuh>
#include <vector>

namespace HugeCTR {

template <typename KeyType>
struct KeyPair {
  KeyType key;
  int feature_id;
  HOST_DEVICE_INLINE KeyType store_idx() const { return (key | 0x1); }
  HOST_DEVICE_INLINE KeyType flatten_idx() const { return (feature_id << 1U | (key & 0x1)); }
  template <typename TableValue>
  HOST_DEVICE_INLINE bool match(const TableValue insert_value) const {
    return ((key & 0x1) == (insert_value.detail.feature_id_and_key_lo & 0x1)) &&
           (feature_id == insert_value.detail.feature_id_and_key_lo >> 1U);
  }
};

struct Hash {
  HOST_DEVICE_INLINE size_t operator()(const KeyPair<uint32_t> &key_pair) {
    using hash_func = MurmurHash3_32<uint32_t>;
    uint32_t key_hash = hash_func::hash(key_pair.key);
    uint32_t feature_id_hash = hash_func::hash(key_pair.feature_id);
    return hash_func ::hash_combine(key_hash, feature_id_hash);
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyPair<int32_t> &key_pair) {
    using hash_func = MurmurHash3_32<int32_t>;
    uint32_t key_hash = hash_func::hash(key_pair.key);
    uint32_t feature_id_hash = hash_func::hash(key_pair.feature_id);
    return hash_func ::hash_combine(key_hash, feature_id_hash);
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyPair<uint64_t> &key_pair) {
    using hash_func = MurmurHash3_32<uint64_t>;
    uint32_t key_hash = hash_func::hash(key_pair.key);
    uint32_t feature_id_hash = hash_func::hash(key_pair.feature_id);
    return hash_func ::hash_combine(key_hash, feature_id_hash);
  }
  HOST_DEVICE_INLINE size_t operator()(const KeyPair<int64_t> &key_pair) {
    using hash_func = MurmurHash3_32<int64_t>;
    uint32_t key_hash = hash_func::hash(key_pair.key);
    uint32_t feature_id_hash = hash_func::hash(key_pair.feature_id);
    return hash_func ::hash_combine(key_hash, feature_id_hash);
  }
};

struct ShardPartitionerView {
  int *gpu_ids;
  int *num_shard_range;

  template <typename KeyType>
  DEVICE_INLINE int operator()(const KeyPair<KeyType> &key_pair) const noexcept {
    const auto &key = key_pair.key;
    const int &feature_id = key_pair.feature_id;
    int num_shard = num_shard_range[feature_id + 1] - num_shard_range[feature_id];
    uint64_t shard_id = (uint64_t)key % (uint64_t)num_shard;
    return gpu_ids[num_shard_range[feature_id] + shard_id];
  }
};

struct ShardPartitioner {
  core23::Tensor gpu_ids;
  core23::Tensor num_shard_range;

  ShardPartitioner() = default;

  ShardPartitioner(std::shared_ptr<core::CoreResourceManager> core,
                   const std::vector<embedding::LookupParam> &lookup_params,
                   const std::vector<std::vector<int>> &shard_matrix,
                   const std::vector<int> &lookup_ids);

  using view_type = ShardPartitionerView;

  view_type view() const noexcept {
    return view_type{gpu_ids.data<int>(), num_shard_range.data<int>()};
  }
};

struct TablePartitionerView {
  int *lookup_id_to_local_table_id;

  template <typename KeyType>
  DEVICE_INLINE int operator()(const KeyPair<KeyType> &key_pair) const noexcept {
    return lookup_id_to_local_table_id[key_pair.feature_id];
  }
};

struct TablePartitioner {
  core23::Tensor lookup_id_to_local_table_id;

  TablePartitioner() = default;

  TablePartitioner(std::shared_ptr<core::CoreResourceManager> core, int num_lookup,
                   const std::vector<int> &local_lookup_id_to_global_lookup_ids,
                   const embedding::WgradAttr &wgrad_attr);
  using view_type = TablePartitionerView;

  view_type view() const noexcept { return view_type{lookup_id_to_local_table_id.data<int>()}; }
};

struct IdentityPartitionerView {
  template <typename KeyType>
  DEVICE_INLINE int operator()(const KeyPair<KeyType> &key_pair) const noexcept {
    return key_pair.feature_id;
  }
};

union TableValue {
  uint64_t value;
  struct Detail {
    uint32_t r_idx_plus_one;
    uint32_t feature_id_and_key_lo;
  } detail;
  template <typename KeyEntry>
  HOST_DEVICE_INLINE void write(uint32_t reverse_idx, KeyEntry key) {
    detail.r_idx_plus_one = reverse_idx;
    detail.feature_id_and_key_lo = (uint32_t)key.flatten_idx();
  }
  HOST_DEVICE_INLINE uint32_t reverse_idx() const { return detail.r_idx_plus_one; }
};

template <typename KeyType>
struct TableEntry {
  using key_type = KeyType;
  using value_type = TableValue;
  KeyType key;
  TableValue value;
};

template <typename KeyType, typename BucketRangeType, typename Partitioner>
struct UniqueTableView {
  TableEntry<KeyType> *table;
  size_t capacity;

  Partitioner partitioner;

  using ResultType = PartitionedDataView<KeyType, BucketRangeType>;

  DEVICE_INLINE uint32_t find(const KeyPair<KeyType> &key_pair, ResultType &result) noexcept {
    int partition_id = partitioner(key_pair);

    KeyType *current_partitioned_keys =
        result.partitioned_keys + partition_id * result.max_num_key_per_partition;
    int *current_feature_ids = result.feature_ids + partition_id * result.max_num_key_per_partition;
    BucketRangeType *current_d_num_key = result.d_num_key_per_partition + partition_id;

    KeyPair<KeyType> unique_out = {0, 0};
    bool is_unique = true;
    auto r_idx_plus_one =
        core23::get_insert_dump<KeyPair<KeyType>, TableEntry<KeyType>, BucketRangeType, Hash>(
            key_pair, table, current_d_num_key, unique_out, capacity, {0, 0}, is_unique);
    if (is_unique) {
      current_partitioned_keys[r_idx_plus_one - 1] = unique_out.key;
      current_feature_ids[r_idx_plus_one - 1] = unique_out.feature_id;
    }
    return partition_id * result.max_num_key_per_partition + r_idx_plus_one;
  }
};
}  // namespace HugeCTR
