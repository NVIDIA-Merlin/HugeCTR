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
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <core23/tensor_params.hpp>

namespace HugeCTR {
namespace core23 {

ncclDataType_t get_nccl_dtype_from_tensor_scalar_type_core23(core23::ScalarType scalar_type);
}
}  // namespace HugeCTR

namespace embedding {
namespace core23 = HugeCTR::core23;
using core::CoreResourceManager;

class NcclAll2AllComm {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  NcclAll2AllComm() = default;

  NcclAll2AllComm(std::shared_ptr<CoreResourceManager> core);

  void communicate(const std::vector<core23::Tensor> &send_tensors,
                   std::vector<core23::Tensor> &recv_tensors);
};

class NcclAllReduceInplaceComm {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  NcclAllReduceInplaceComm() = default;

  NcclAllReduceInplaceComm(std::shared_ptr<CoreResourceManager> core);

  void communicate(core23::Tensor &tensors, size_t count);
};

}  // namespace embedding
