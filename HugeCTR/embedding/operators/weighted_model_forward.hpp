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

class WeightedModelForward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_local_embedding_;

 public:
  WeightedModelForward() = default;

  WeightedModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                       const std::vector<int> &local_embedding_list);

  void compute(const core23::Tensor &mp_ev, const core23::Tensor &model_offset,
               core23::Tensor &model_comm_buffer, const core23::Tensor &d_local_ev_size_list,
               const core23::Tensor &d_local_ev_size_offset, int batch_size, int max_ev_size,
               const core23::Tensor &sp_weight);
};

}  // namespace embedding
