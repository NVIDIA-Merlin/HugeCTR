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
using core::Device;
using core::Tensor;
using core::TensorList;

class NetworkBackward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;

 public:
  NetworkBackward() = default;

  NetworkBackward(std::shared_ptr<CoreResourceManager> core, int num_gpus)
      : core_(core), num_gpus_(num_gpus) {}

  void compute(const Tensor& top_grad, const Tensor& d_ev_size_offset, const Tensor& gpu_idx_offset,
               const TensorList& global_ev_offset, const Tensor& network_idx,
               const Tensor& network_offset, const Tensor& network_dst,
               TensorList& network_comm_buffer, int batch_size);
};

}  // namespace embedding
