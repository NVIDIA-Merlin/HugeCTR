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

#include <embedding/operators/communication.hpp>
#include <utils.hpp>

namespace embedding {

using core::CoreResourceManager;
using core::get_nccl_dtype_from_tensor_scalar_type;

NcclAll2AllComm::NcclAll2AllComm(std::shared_ptr<CoreResourceManager> core) : core_(core) {}

void NcclAll2AllComm::communicate(const std::vector<Tensor> &send_tensors,
                                  const std::vector<size_t> &send_offsets,
                                  std::vector<Tensor> &recv_tensors,
                                  const std::vector<size_t> &recv_offsets) {
  int device_id = core_->get_device_id();
  auto &comm = core_->get_nccl();

  HugeCTR::CudaDeviceContext ctx(device_id);
  HCTR_LIB_THROW(ncclGroupStart());
  int num_total_gpu = core_->get_global_gpu_count();
  for (int p = 0; p < num_total_gpu; ++p) {
    ncclDataType_t nccl_dtype =
        get_nccl_dtype_from_tensor_scalar_type(send_tensors[p].dtype().type());
    HCTR_LIB_THROW(ncclSend(send_tensors[p].get(), send_offsets[p], nccl_dtype, p, comm,
                            core_->get_local_gpu()->get_stream()));
    HCTR_LIB_THROW(ncclRecv(recv_tensors[p].get(), recv_offsets[p], nccl_dtype, p, comm,
                            core_->get_local_gpu()->get_stream()));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
}

NcclAllReduceInplaceComm::NcclAllReduceInplaceComm(std::shared_ptr<CoreResourceManager> core)
    : core_(core) {}

void NcclAllReduceInplaceComm::communicate(Tensor &tensor, size_t count) {
  int device_id = core_->get_device_id();
  HugeCTR::CudaDeviceContext ctx(device_id);
  ncclDataType_t nccl_dtype = get_nccl_dtype_from_tensor_scalar_type(tensor.dtype().type());

  HCTR_LIB_THROW(ncclAllReduce(tensor.get(), tensor.get(), count, nccl_dtype, ncclSum,
                               core_->get_nccl(), core_->get_local_gpu()->get_stream()));
}

}  // namespace embedding