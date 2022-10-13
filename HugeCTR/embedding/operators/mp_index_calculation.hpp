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

#include "HugeCTR/core/buffer.hpp"
#include "HugeCTR/core/registry.hpp"
#include "embedding/common.hpp"
namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::DeviceType;
using core::Shape;
using core::Tensor;
using core::TensorList;
using core::TensorScalarType;

class ModelIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

  int num_local_embedding_;
  int local_hotness_sum_;
  int hotness_list_sum_;
  int universal_batch_size_;
  Tensor model_key_;
  Tensor model_idx_offsets_;
  Tensor num_key_in_bucket_for_combiner_;
  Tensor num_model_key_;
  Tensor flag_;

  Tensor d_temp_scan_storage_;
  Tensor d_temp_select_storage_;

 public:
  ModelIndexCalculation() = default;

  ModelIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_local_embedding,
                        int local_hotness_sum, int hotness_sum, int universal_batch_size,
                        DataType key_type);

  void compute(const Tensor& key, const Tensor& bucket_range, size_t num_key,
               const Tensor& d_local_embedding_list, const Tensor& d_local_shard_id_list,
               const Tensor& d_local_num_shards_list, int batch_size, Tensor* model_key,
               Tensor* model_idx_offsets, size_t* num_model_key);
};

class ModelBackwardIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

  int num_gpus_;
  int num_local_embedding_;
  size_t sort_end_bit_;

  Tensor bucket_id_list_;

  Tensor hash_keys_;
  Tensor hash_offset_;
  Tensor local_index_;

  Tensor sorted_local_index_;
  Tensor unique_local_index_;

  Tensor unique_key_;
  Tensor num_unique_key_;
  Tensor unique_dst_idx_;
  Tensor sorted_bucket_id_list_;
  Tensor sorted_bucket_id_offset_;
  Tensor unique_id_space_offset_;
  Tensor coordinate_wgrad_dst_idx_;

  Tensor unique_id_space_list_;
  Tensor unique_id_space_ev_size_list_;

  Tensor d_temp_sort_storage_;
  Tensor d_temp_run_length_encode_storage_;
  Tensor d_temp_scan_encode_storage_;

 public:
  ModelBackwardIndexCalculation() = default;

  ModelBackwardIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                int num_local_embedding,
                                const std::vector<int>& h_local_hotness_list,
                                const std::vector<int>& h_local_id_space_list,
                                const std::vector<int>& h_local_ev_size_list,
                                int universal_batch_size, DataType key_type);

  void compute(const Tensor& model_key, size_t num_model_key, const Tensor& model_offset,
               const Tensor& id_space_offset, const Tensor& id_space_list, int batch_size,
               Tensor* unique_key, size_t* num_unique_key, Tensor* unique_dst_idx,
               Tensor* sorted_bucket_id_list, Tensor* sorted_bucket_id_offset,
               Tensor* unique_id_space_list, Tensor* unique_id_space_offset, Tensor* coordinate_key,
               Tensor* coordinate_wgrad_dst_idx);
};
}  // namespace embedding
