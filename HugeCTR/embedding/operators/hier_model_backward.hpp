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

namespace embedding {

struct IntraModelCommBufferAttr;
struct IntraModelReductionBuffer;
struct ModelCommBuffer;

struct IntraModelBackward {
  std::shared_ptr<CoreResourceManager> core_;

  void backward(const IntraModelCommBufferAttr &intra_model_comm_buffer_attr,
                const IntraModelReductionBuffer &reduction_buffer,
                ModelCommBuffer &model_comm_buffer, int batch_size);
};
}  // namespace embedding
