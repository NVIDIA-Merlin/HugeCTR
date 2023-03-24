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

namespace HugeCTR {
namespace core23 {

ncclDataType_t get_nccl_dtype_from_tensor_scalar_type_core23(core23::ScalarType scalar_type) {
  switch (scalar_type) {
    case core23::ScalarType::Float:
      return ncclFloat32;
    case core23::ScalarType::Half:
      return ncclHalf;
    case core23::ScalarType::Int64:
      return ncclInt64;
    case core23::ScalarType::UInt64:
      return ncclUint64;
    case core23::ScalarType::Int32:
      return ncclInt32;
    case core23::ScalarType::UInt32:
      return ncclUint32;
    case core23::ScalarType::Char:
      return ncclChar;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                     "Not supported core23::ScalarType to NcclDataType_t");
  }
  return ncclInt;
}
}  // namespace core23
}  // namespace HugeCTR

namespace embedding {
NcclAll2AllComm::NcclAll2AllComm(std::shared_ptr<CoreResourceManager> core) : core_(core) {}

void NcclAll2AllComm::communicate(const std::vector<core23::Tensor>& send_tensors,
                                  std::vector<core23::Tensor>& recv_tensors) {
  int device_id = core_->get_device_id();
  auto& comm = core_->get_nccl();

  HugeCTR::CudaDeviceContext ctx(device_id);
  HCTR_LIB_THROW(ncclGroupStart());
  int num_total_gpu = core_->get_global_gpu_count();
  for (int p = 0; p < num_total_gpu; ++p) {
    ncclDataType_t nccl_dtype =
        core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensors[p].data_type().type());
    HCTR_LIB_THROW(ncclSend(send_tensors[p].data(), send_tensors[p].num_elements(), nccl_dtype, p,
                            comm, core_->get_local_gpu()->get_stream()));
    HCTR_LIB_THROW(ncclRecv(recv_tensors[p].data(), recv_tensors[p].num_elements(), nccl_dtype, p,
                            comm, core_->get_local_gpu()->get_stream()));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
}

void NcclAll2AllComm::hier_communicate(const std::vector<core23::Tensor>& send_tensors,
                                       std::vector<core23::Tensor>& recv_tensors) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  auto& comm = core_->get_nccl();
  auto stream = core_->get_local_gpu()->get_stream();

  int num_total_gpu = core_->get_global_gpu_count();
  int num_local_gpu = core_->get_local_gpu_count();
  int local_gpu_id = core_->get_local_gpu_id();
  int num_node = num_total_gpu / num_local_gpu;
  HCTR_CHECK(send_tensors.size() == static_cast<size_t>(num_node));
  HCTR_CHECK(recv_tensors.size() == static_cast<size_t>(num_node));

  HCTR_LIB_THROW(ncclGroupStart());
  for (int node_id = 0; node_id < num_node; ++node_id) {
    ncclDataType_t nccl_dtype = core23::get_nccl_dtype_from_tensor_scalar_type_core23(
        send_tensors[node_id].data_type().type());
    HCTR_LIB_THROW(ncclSend(send_tensors[node_id].data(), send_tensors[node_id].num_elements(),
                            nccl_dtype, node_id * num_local_gpu + local_gpu_id, comm, stream));
    HCTR_LIB_THROW(ncclRecv(recv_tensors[node_id].data(), recv_tensors[node_id].num_elements(),
                            nccl_dtype, node_id * num_local_gpu + local_gpu_id, comm, stream));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
}

NcclAllReduceInplaceComm::NcclAllReduceInplaceComm(std::shared_ptr<CoreResourceManager> core)
    : core_(core) {}

void NcclAllReduceInplaceComm::communicate(core23::Tensor& tensor, size_t count) {
  int device_id = core_->get_device_id();
  HugeCTR::CudaDeviceContext ctx(device_id);
  ncclDataType_t nccl_dtype =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(tensor.data_type().type());

  HCTR_LIB_THROW(ncclAllReduce(tensor.data(), tensor.data(), count, nccl_dtype, ncclSum,
                               core_->get_nccl(), core_->get_local_gpu()->get_stream()));
}

}  // namespace embedding
