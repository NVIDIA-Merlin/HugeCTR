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

namespace embedding {
using core::CoreResourceManager;
using core::Device;
using core::Tensor;

class NcclAll2AllComm {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  NcclAll2AllComm() = default;

  NcclAll2AllComm(std::shared_ptr<CoreResourceManager> core);

  void communicate(const std::vector<Tensor> &send_tensors, const std::vector<size_t> &send_offsets,
                   std::vector<Tensor> &recv_tensors, const std::vector<size_t> &recv_offsets);
};

class NcclAllReduceInplaceComm {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  NcclAllReduceInplaceComm() = default;

  NcclAllReduceInplaceComm(std::shared_ptr<CoreResourceManager> core);

  void communicate(Tensor &tensors, size_t count);
};

}  // namespace embedding