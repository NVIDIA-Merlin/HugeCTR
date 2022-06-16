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

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::DeviceType;
using core::Shape;
using core::Tensor;
using core::TensorList;
using HugeCTR::CudaDeviceContext;
using HugeCTR::TensorScalarType;

class DPIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  DPIndexCalculation() = default;

  DPIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                     int num_local_embedding, int local_hotness_sum, int hotness_sum,
                     int universal_batch_size, DataType key_type, DataType offset_type);

  void compute(const Tensor& key, const Tensor& bucket_range, size_t num_keys,
               const Tensor& d_local_embedding_list, int batch_size, Tensor* ret_dp_key,
               Tensor* ret_dp_offset, size_t* num_dp_key, Tensor* dp_dst);

 private:
  int num_gpus_;
  int num_local_embedding_;
  int universal_batch_size_;
  int universal_batch_size_per_gpu_;
  int local_hotness_sum_;
  int hotness_sum_;
  DataType key_type_;
  DataType offset_type_;

  Tensor num_dp_key_;
  Tensor flag_;

  Tensor d_temp_storage_category_;
  Tensor d_temp_storage_offset_;

  // outputs
  Tensor dp_key_;
  Tensor dp_offset_;
  Tensor dp_dst_;
};

class DPLocalReduceIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;

  int num_local_embedding_;
  int num_embedding_;

  Tensor segment_start_offsets_;
  Tensor segment_end_offsets_;
  Tensor dp_keys_;
  Tensor dp_bucket_id_;
  Tensor sorted_dp_keys_;
  Tensor sorted_dp_bucket_id_;

  Tensor unique_dp_keys_;
  Tensor unique_dp_keys_indices_;
  Tensor num_unique_key_;
  Tensor sorted_bucket_id_list_;
  Tensor num_sorted_bucket_id_;
  Tensor unique_dst_idx_;
  Tensor sorted_bucket_id_offset_;
  Tensor unique_id_space_offset_;

  Tensor d_temp_segmented_sort_storage_;
  Tensor d_temp_if_storage_;
  Tensor d_temp_select_bucket_id_storage_;
  Tensor d_scan_storage_;

 public:
  DPLocalReduceIndexCalculation() = default;

  DPLocalReduceIndexCalculation(std::shared_ptr<CoreResourceManager> core, int num_embedding_,
                                int num_local_embedding,
                                const std::vector<int>& h_local_hotness_list,
                                int universal_batch_size, DataType key_type);

  void compute(const Tensor& key, size_t num_key, const Tensor& bucket_range,
               const Tensor& d_local_embedding_list, const Tensor& id_space_list,
               const Tensor& d_ev_size_list, int batch_size, Tensor* unique_key,
               size_t* num_unique_key, Tensor* unique_dst_idx, Tensor* sorted_bucket_id_list,
               Tensor* sorted_bucket_id_offset, Tensor* unique_id_space_offset);
};
}  // namespace embedding
