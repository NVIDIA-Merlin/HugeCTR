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

#include <core23/registry.hpp>
#include <embedding/common.hpp>

namespace embedding {

namespace core23 = HugeCTR::core23;
using core::CoreResourceManager;

class WeightedModelIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

  int num_local_embedding_;
  int local_hotness_sum_;
  int hotness_list_sum_;
  int universal_batch_size_;
  core23::Tensor model_key_;
  core23::Tensor model_sp_weight_;
  core23::Tensor model_idx_offsets_;
  core23::Tensor num_key_in_bucket_for_combiner_;
  core23::Tensor num_model_key_;
  core23::Tensor flag_;

  core23::Tensor d_temp_scan_storage_;
  core23::Tensor d_temp_select_storage_;
  core23::Tensor d_temp_select_weight_storage_;

 public:
  WeightedModelIndexCalculation() = default;

  WeightedModelIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_local_embedding,
                                int local_hotness_sum, int hotness_sum, int universal_batch_size,
                                core23::DataType key_type);

  void compute(const core23::Tensor& key, const core23::Tensor& bucket_range, size_t num_key,
               const core23::Tensor& d_local_embedding_list,
               const core23::Tensor& d_local_shard_id_list,
               const core23::Tensor& d_local_num_shards_list, int batch_size,
               core23::Tensor& model_key, core23::Tensor& model_idx_offsets, size_t* num_model_key,
               const core23::Tensor& reorder_sp_weight, core23::Tensor& model_sp_weight);
};

class WeightedModelBackwardIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

  int num_gpus_;
  int num_local_embedding_;
  size_t sort_end_bit_;

  core23::Tensor bucket_id_list_;

  core23::Tensor hash_keys_;
  core23::Tensor hash_offset_;
  core23::Tensor local_index_;

  core23::Tensor sorted_local_index_;
  core23::Tensor unique_local_index_;

  core23::Tensor unique_key_;
  core23::Tensor num_unique_key_;
  core23::Tensor unique_dst_idx_;
  core23::Tensor sorted_bucket_id_list_;
  core23::Tensor sorted_bucket_id_offset_;
  core23::Tensor unique_id_space_offset_;
  core23::Tensor coordinate_wgrad_dst_idx_;

  core23::Tensor unique_id_space_list_;
  core23::Tensor unique_id_space_ev_size_list_;

  core23::Tensor d_temp_sort_storage_;
  core23::Tensor d_temp_sort_sp_weight_storage_;
  core23::Tensor d_temp_sort_sp_weight_key_;
  core23::Tensor sorted_sp_weight_list_;
  core23::Tensor d_temp_run_length_encode_storage_;
  core23::Tensor d_temp_scan_encode_storage_;

 public:
  WeightedModelBackwardIndexCalculation() = default;

  WeightedModelBackwardIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                        int num_local_embedding,
                                        const std::vector<int>& h_local_hotness_list,
                                        const std::vector<int>& h_local_id_space_list,
                                        const std::vector<int>& h_local_ev_size_list,
                                        int universal_batch_size, core23::DataType key_type);

  void compute(const core23::Tensor& model_key, size_t num_model_key,
               const core23::Tensor& model_offset, const core23::Tensor& id_space_offset,
               const core23::Tensor& id_space_list, int batch_size, core23::Tensor& unique_key,
               uint64_t* num_unique_key, core23::Tensor& unique_dst_idx,
               core23::Tensor& sorted_bucket_id_list, core23::Tensor& sorted_bucket_id_offset,
               core23::Tensor& unique_id_space_list, core23::Tensor& unique_id_space_offset,
               core23::Tensor& coordinate_key, core23::Tensor& coordinate_wgrad_dst_idx,
               const core23::Tensor& model_sp_weight, core23::Tensor& coordinate_sp_weight);
};
}  // namespace embedding
