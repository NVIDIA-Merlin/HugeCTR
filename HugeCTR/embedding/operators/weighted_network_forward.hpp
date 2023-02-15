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

#include <core/buffer.hpp>
#include <core/registry.hpp>
#include <embedding/common.hpp>

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::Shape;
using core::Tensor;
using core::TensorList;

class WeightedNetworkForward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;

 public:
  WeightedNetworkForward() = default;

  WeightedNetworkForward(std::shared_ptr<CoreResourceManager> core, int num_gpus);

  void compute(const TensorList& row_lengths, const Tensor& d_combiner_list,
               const TensorList& network_comm_buffer, const Tensor& network_ids,
               const Tensor& network_gpu_ids, const Tensor& network_offsets,
               const Tensor& network_dst_lookup_ids, const TensorList& network_ev_sizes,
               const TensorList& network_ev_offsets, TensorList& output_buffer,
               const Tensor& d_ev_size_offset, int batch_size, int max_ev_size,
               const Tensor& sp_weight_sum);
};

}  // namespace embedding
