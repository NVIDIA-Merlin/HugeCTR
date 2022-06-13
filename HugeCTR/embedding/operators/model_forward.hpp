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
#include "HugeCTR/core/registry.hpp"
#include "HugeCTR/embedding/common.hpp"

namespace embedding {

class DPModelForward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_embedding_;
  int num_local_embedding_;

 public:
  DPModelForward() = default;

  DPModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus, int num_embedding,
                 int num_local_embedding);

  void compute(const TensorList &dp_ev, const Tensor &dp_offset, const Tensor &dp_dst,
               Tensor &output_buffer, const Tensor &d_local_ev_size_list,
               const Tensor &d_local_combiner_list, const Tensor &d_ev_size_offset,
               int batch_size) const;
};

class ModelForward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_local_embedding_;

 public:
  ModelForward() = default;

  ModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
               const std::vector<int> &local_embedding_list);

  void compute(const TensorList &mp_ev, const Tensor &model_offset, TensorList &model_comm_buffer,
               const Tensor &d_local_ev_size_list, const Tensor &d_local_ev_size_offset,
               int batch_size);
};
}  // namespace embedding
