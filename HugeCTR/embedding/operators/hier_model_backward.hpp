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

#include <HugeCTR/core/core.hpp>

#include "../embedding_table.hpp"
#include "network_forward.hpp"

namespace embedding {

struct IntraModelCommBufferAttr;
struct IntraModelReductionBuffer;
struct ModelCommBuffer;

struct IntraModelBackwardAttr {
  NetworkIndices indices;
  std::vector<int> h_global_lookup_ids_in_local_gpu;
  core23::Tensor global_lookup_ids_in_local_gpu;

  std::vector<int> h_evsizes_in_local_gpu;
  core23::Tensor evsizes_in_local_gpu;

  std::vector<int> h_local_gpu_lookup_ids_to_node_lookup_ids;
  core23::Tensor local_gpu_lookup_ids_to_node_lookup_ids;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id, const std::vector<std::vector<int>> &h_lookup_ids_in_current_node);
};

struct IntraModelBackward {
  std::shared_ptr<CoreResourceManager> core_;
  IntraModelBackwardAttr attr;
  void backward(const IntraModelCommBufferAttr &intra_model_comm_buffer_attr,
                const IntraModelReductionBuffer &reduction_buffer,
                const EmbeddingInput &embedding_input, ModelCommBuffer &model_comm_buffer,
                int batch_size);
};
}  // namespace embedding
