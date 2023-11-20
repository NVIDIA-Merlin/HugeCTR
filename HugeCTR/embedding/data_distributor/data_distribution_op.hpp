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

#include <core/core.hpp>
#include <embedding/common.hpp>
#include <embedding/data_distributor/data_compression_operators.hpp>
#include <embedding/data_distributor/key_filtering_operators.hpp>
#include <embedding/operators/compress_offset.hpp>
#include <embedding/operators/keys_to_indices.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

struct ShardPartitioner;
struct TablePartitioner;

class IDataDistributionOp {
 public:
  virtual ~IDataDistributionOp() = default;

  virtual void distribute(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                          int batch_size, cudaStream_t stream) = 0;
};

class DenseMPDataDistributionOp final : public IDataDistributionOp {
 public:
  DenseMPDataDistributionOp(
      std::shared_ptr<core::CoreResourceManager> core,
      const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id,
      const std::vector<embedding::EmbeddingTableParam> &emb_table_param_list);

  void distribute(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                  int batch_size, cudaStream_t stream) override;

  void filter_before_all2all(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                             int batch_size, cudaStream_t stream);

  void all2all_keys_per_bucket(embedding::EmbeddingInput &output, cudaStream_t stream);

  void all2all_keys(embedding::EmbeddingInput &output, cudaStream_t stream);

  void filter_after_all2all(embedding::EmbeddingInput &output, cudaStream_t stream);

  void convert_indices(embedding::EmbeddingInput &output);

 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  embedding::EmbeddingCollectionParam ebc_param_;

  size_t num_global_gpus_;
  // Dense Unique Operators
  struct DenseMPTempStorage {
    DenseMPTempStorage(std::shared_ptr<core::CoreResourceManager> core,
                       const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id);
    int num_table;

    std::unique_ptr<ShardPartitioner> shard_partitioner_;
    std::unique_ptr<TablePartitioner> table_partitioner_;

    core23::Tensor h_num_network_reverse_idx;  // uint64_t
    PartitionedData partitioned_data_after_shard_matrix_partition;
    core23::Tensor d_num_key_gpu_major;    // bucket_range_type
    core23::Tensor keys_gpu_major;         // kDenseDPTempStorageey_type
    core23::Tensor feature_ids_gpu_major;  // int

    PartitionedData partitioned_data_after_table_id_partition;
  } dense_temp_storage_;

  PartitionAndUniqueOperator partition_and_unique_operator_;
  CompressReverseIdxRangeOperator compress_reverse_idx_range_operator_;
  CompactPartitionDataOperator compact_partitioned_data_operator_;

  std::unique_ptr<embedding::KeysToIndicesConverter> indices_converter_;
  bool do_reduction_;
};

class SparseMPDataDistributionOp : public IDataDistributionOp {
 public:
  SparseMPDataDistributionOp(
      std::shared_ptr<core::CoreResourceManager> core,
      const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id,
      const std::vector<embedding::EmbeddingTableParam> &emb_table_param_list);

  void distribute(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                  int batch_size, cudaStream_t stream) override;

  void filter_before_all2all(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                             cudaStream_t stream);

  void all2all_keys_per_bucket(embedding::EmbeddingInput &output, cudaStream_t stream);

  void all2all_keys(embedding::EmbeddingInput &output, cudaStream_t stream);

  void filter_after_all2all(embedding::EmbeddingInput &output, cudaStream_t stream);

  void convert_indices(embedding::EmbeddingInput &output);

 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  embedding::EmbeddingCollectionParam ebc_param_;
  size_t num_global_gpus_;
  size_t batch_size_per_gpu_;
  size_t sample_max_nnz_;

  struct MPTempStorage {
    MPTempStorage() = default;

    MPTempStorage(std::shared_ptr<core::CoreResourceManager> core, int batch_size,
                  int sample_max_nnz, int max_local_features, int max_local_buckets,
                  int max_buckets_in_group, core23::DataType key_type,
                  core23::DataType offset_type);

    core23::Tensor temp_sort_storage;
    core23::Tensor temp_scan_storage;
    core23::Tensor k_per_b_gpu_major;       // keys-per-bucket
    core23::Tensor k_per_g;                 // keys-per-gpu
    core23::Tensor bucket_range_gpu_major;  // received from nccl
    core23::Tensor sorted_local_keys;
    core23::Tensor sorted_local_labels;
    core23::Tensor keys;                // received from nccl
    core23::Tensor k_per_b_feat_major;  // keys-per-bucket
    void *h_send_k_per_g;
    void *h_recv_k_per_g;
  } sparse_temp_storage_;

  mp::LabelAndCountKeysOperator label_and_count_keys_operator_;
  mp::LabelAndCountKeysOperator::Result label_and_count_keys_output_;
  mp::CountKeysOperator count_keys_operator_;
  mp::TransposeBucketsOperator transpose_buckets_operator_;
  mp::SwizzleKeysOperator swizzle_keys_operator_;

  core23::Tensor d_local_table_ids_;
  std::unique_ptr<embedding::CompressOffset> compress_offset_;
  std::unique_ptr<embedding::KeysToIndicesConverter> indices_converter_;
};

class SparseDPDataDistributionOp final : public IDataDistributionOp {
 public:
  SparseDPDataDistributionOp(
      std::shared_ptr<core::CoreResourceManager> core,
      const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id,
      const std::vector<embedding::EmbeddingTableParam> &emb_table_param_list);

  void distribute(const DataDistributionInput &input, embedding::EmbeddingInput &output,
                  int batch_size, cudaStream_t stream) override;

  void convert_indices(embedding::EmbeddingInput &output);

 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  embedding::EmbeddingCollectionParam ebc_param_;
  size_t batch_size_per_gpu_;

  dp::ConcatKeysAndBucketRangeOperator concat_keys_and_bucket_range_operator_;

  core23::Tensor d_local_table_ids_;
  std::unique_ptr<embedding::CompressOffset> compress_offset_;
  std::unique_ptr<embedding::KeysToIndicesConverter> indices_converter_;
};
}  // namespace HugeCTR
