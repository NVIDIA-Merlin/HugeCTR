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
#include <embedding/data_distributor/data_compression_operators.hpp>
#include <embedding/data_distributor/data_distribution_op.hpp>
#include <embedding/data_distributor/key_filtering_operators.hpp>
#include <embedding/operators/compress_offset.hpp>
#include <embedding/operators/dp_index_calculation.hpp>
#include <embedding/operators/keys_to_indices.hpp>
#include <embedding/operators/mp_index_calculation.hpp>
#include <embedding/operators/transpose_input.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

namespace HugeCTR {
class DataDistributor {
 public:
  using Result = std::vector<embedding::EmbeddingInput>;

  DataDistributor(std::vector<std::shared_ptr<core::CoreResourceManager>>& core_resource_managers,
                  const embedding::EmbeddingCollectionParam& ebc_param,
                  const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list,
                  const std::vector<int>& dr_lookup_ids);

  void distribute(int gpu_id, const std::vector<core23::Tensor>& dp_keys,
                  const std::vector<core23::Tensor>& dp_bucket_range, Result& output,
                  int batch_size);

 private:
  struct GpuCommData {
    // This is a performance optimization to prevent us from computing bucket ranges each iteration.
    // If the current_batch_size == last_batch_size then the bucket_ranges are the same.
    int last_batch_size;
    core23::Tensor hotness_bucket_range;
  };

  void init_comm_data();

  void init_filtered_all_to_all();

  void init_fixed_dp_bucket_range();

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_managers_;
  std::vector<int> feature_pooling_factors_;
  std::vector<std::vector<int>> resident_feature_tables_;  // [gpu_id][feature_id]
  std::vector<GpuCommData> gpu_comm_data_;

  std::vector<DataDistributionInput> data_distribution_input_;
  std::vector<std::vector<std::unique_ptr<IDataDistributionOp>>> data_distribution_ops_;

  // Key Filtering (MP)
  std::vector<ComputeDPBucketRangeOperator> compute_dp_bucket_range_operators_;

  std::vector<std::vector<core23::Tensor>> fixed_dp_bucket_range_;

  size_t batch_size_;
  size_t batch_size_per_gpu_;

  embedding::EmbeddingCollectionParam ebc_param_;
  std::unordered_map<size_t, size_t> feature_id_to_group_id_map_;
  std::unordered_map<size_t, size_t> feature_id_to_table_id_map_;

  std::vector<embedding::EmbeddingTableParam> emb_table_param_list_;

  size_t num_local_gpus_;
  size_t num_global_gpus_;
  size_t num_features_;
};

DataDistributor::Result allocate_output_for_data_distributor(
    std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param);
}  // namespace HugeCTR
