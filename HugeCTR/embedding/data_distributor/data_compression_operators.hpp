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
#include <embedding/common.hpp>
#include <memory>
#include <vector>

namespace HugeCTR {

template <typename KeyType, typename BucketRangeType>
struct PartitionedDataView {
  KeyType *partitioned_keys;
  int *feature_ids;
  BucketRangeType *d_num_key_per_partition;
  size_t max_num_key_per_partition;
};

template <typename KeyType, typename BucketRangeType>
struct CompressedDataView {
  PartitionedDataView<KeyType, BucketRangeType> partitioned_data;
  BucketRangeType *reverse_idx;
};

struct PartitionedData {
  core23::Tensor partitioned_keys;  // KeyType. num freq key or num infreq key or num unique key
  core23::Tensor feature_ids;       // int32_t. num freq key or num infreq key or num unique key
  core23::Tensor d_num_key_per_partition;  // bucket_range_type

  size_t max_num_key_per_partition;  // stride

  PartitionedData() = default;

  PartitionedData(std::shared_ptr<core::CoreResourceManager> core, size_t num_partition,
                  size_t max_num_key_per_partition, core23::DataType key_type,
                  core23::DataType bucket_range_type);

  template <typename KeyType, typename BucketRangeType>
  PartitionedDataView<KeyType, BucketRangeType> view() {
    return PartitionedDataView<KeyType, BucketRangeType>{
        partitioned_keys.data<KeyType>(), feature_ids.data<int>(),
        d_num_key_per_partition.data<BucketRangeType>(), max_num_key_per_partition};
  }
};

struct CompressedData {
  PartitionedData partitioned_data;

  core23::Tensor reverse_idx;  // length same as non-compressed keys
};

struct CompactedPartitionData {
  core23::Tensor keys;
  core23::Tensor h_num_keys;
  core23::Tensor num_key_per_table;
};

class PartitionAndUniqueOperator {
 public:
  PartitionAndUniqueOperator(std::shared_ptr<core::CoreResourceManager> core,
                             const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id);

  void init_hash_table_for_unique(std::shared_ptr<core::CoreResourceManager> core,
                                  core23::DataType key_type);

  void fill_continuous_bucket_ids(const DataDistributionInput &input, core23::Tensor &bucket_ids,
                                  core23::Tensor &h_num_bucket_ids, int batch_size,
                                  cudaStream_t stream);
  void fill_continuous_bucket_ids_for_reduction(const DataDistributionInput &input,
                                                core23::Tensor &bucket_ids,
                                                core23::Tensor &h_num_bucket_ids, int batch_size,
                                                cudaStream_t stream);
  // dense mp
  template <typename Partitioner>
  void partition_and_unique_on_dp_input(const DataDistributionInput &input,
                                        const Partitioner &shard_partitioner,
                                        CompressedData &compressed_data, cudaStream_t stream);

  // dense mp and infreq
  template <typename Partitioner>
  void partition_and_unique_by_table_id(const core23::Tensor &keys_gpu_major,
                                        const core23::Tensor &feature_ids_gpu_major,
                                        size_t num_keys, const Partitioner &table_partitioner,
                                        CompressedData &compressed_data, cudaStream_t stream);

 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  core23::Tensor frequent_key_hash_table_storage_;

  core23::Tensor hash_table_storage_;
  size_t table_capacity_;

  core23::Tensor range_on_lookup_ids;

  core23::Tensor d_lookup_ids_;  // int
  int num_local_lookup_;
  int num_local_features_;
  int num_features_;
  int batch_size_;
  int global_gpu_count_;
  int batch_size_per_gpu_;
  int last_batch_size_;
};

class CompactPartitionDataOperator {
 public:
  CompactPartitionDataOperator(std::shared_ptr<core::CoreResourceManager> core, int num_table);

  void operator()(const PartitionedData &partitioned_data,
                  CompactedPartitionData &compacted_partition_data, cudaStream_t stream) const;

 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  core23::Tensor d_scan_num_key_per_table_temp_storage;
};

class CompressReverseIdxRangeOperator {
 public:
  explicit CompressReverseIdxRangeOperator(std::shared_ptr<core::CoreResourceManager> core)
      : core_(core) {}

  void operator()(size_t num_bucket_ids, CompressedData &compressed_data,
                  cudaStream_t stream) const;

 private:
  std::shared_ptr<core::CoreResourceManager> core_;
};
}  // namespace HugeCTR
