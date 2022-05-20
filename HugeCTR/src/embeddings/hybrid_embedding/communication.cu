/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "common.hpp"
#include "tensor2.hpp"

namespace {
template <typename T>
ncclDataType_t get_nccl_type();
template <>
ncclDataType_t get_nccl_type<int>() {
  return ncclInt32;
}
template <>
ncclDataType_t get_nccl_type<unsigned int>() {
  return ncclUint32;
}
template <>
ncclDataType_t get_nccl_type<unsigned long long>() {
  return ncclUint64;
}
template <>
ncclDataType_t get_nccl_type<float>() {
  return ncclFloat32;
}
template <>
ncclDataType_t get_nccl_type<__half>() {
  return ncclFloat16;
}
}  // namespace

namespace HugeCTR {

namespace hybrid_embedding {

Communication::Communication(size_t width_data_field) : width_data_field_(width_data_field) {}

/*
 * All to All communications
 */
template <typename commtype>
AllToAllVComm<commtype>::AllToAllVComm(Tensor2<commtype> send_buffer, Tensor2<commtype> recv_buffer,
                                       const uint32_t* send_offsets, const uint32_t* recv_offsets,
                                       const GPUResource* gpu_resource, size_t width_data_field)
    : Communication(width_data_field),
      send_buffer_(send_buffer),
      recv_buffer_(recv_buffer),
      send_offsets_(send_offsets),
      recv_offsets_(recv_offsets),
      gpu_resource_(gpu_resource) {}

template <typename commtype>
void AllToAll_Multi_NCCL<commtype>::communicate(cudaStream_t stream) {
  auto& comm = this->gpu_resource_->get_nccl();
  auto type = get_nccl_type<commtype>();

  int num_global_gpus;
  HCTR_LIB_THROW(ncclCommCount(comm, &num_global_gpus));

  HCTR_LIB_THROW(ncclGroupStart());
  for (int i = 0; i < num_global_gpus; i++) {
    HCTR_LIB_THROW(
        ncclSend(this->send_buffer_.get_ptr() + this->send_offsets_[i] * this->width_data_field_,
                 (this->send_offsets_[i + 1] - this->send_offsets_[i]) * this->width_data_field_,
                 type, i, comm, stream));
    HCTR_LIB_THROW(
        ncclRecv(this->recv_buffer_.get_ptr() + this->recv_offsets_[i] * this->width_data_field_,
                 (this->recv_offsets_[i + 1] - this->recv_offsets_[i]) * this->width_data_field_,
                 type, i, comm, stream));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
}

/*
 * All Reduce communications
 */
template <typename commtype>
AllReduceComm<commtype>::AllReduceComm(AllReduceInPlaceComm* ar_comm,
                                       AllReduceInPlaceComm::Handle ar_handle,
                                       const GPUResource* gpu_resource)
    : Communication(0), ar_comm_(ar_comm), ar_handle_(ar_handle), gpu_resource_(gpu_resource) {}

template <typename commtype>
void AllReduceComm<commtype>::communicate(cudaStream_t stream) {
  ar_comm_->all_reduce(ar_handle_, stream, gpu_resource_->get_local_id());
}

#ifdef ENABLE_MPI
template <typename commtype>
HierAll2Allv_Multi_IB<commtype>::HierAll2Allv_Multi_IB(uint32_t instance_id,
                                                       HierA2AvCollHandle coll_handle,
                                                       size_t** send_sizes,
                                                       const GPUResource* gpu_resource,
                                                       IbComm* ib_comm, cudaStream_t comm_stream)
    : Communication(sizeof(commtype)),
      instance_id_(instance_id),
      coll_handle_(coll_handle),
      send_sizes_(send_sizes),
      gpu_resource_(gpu_resource),
      ib_comm_(ib_comm),
      comm_stream_(comm_stream) {
  HCTR_LIB_THROW(cudaEventCreate(&comm_event_));
}

template <typename commtype>
void HierAll2Allv_Multi_IB<commtype>::update_sizes(cudaStream_t stream) {
  ib_comm_->pre_intra_update_a2a_coll_sizes(coll_handle_, send_sizes_, stream, instance_id_);
}

template <typename commtype>
void HierAll2Allv_Multi_IB<commtype>::communicate(cudaStream_t stream) {
  ib_comm_->post_send_command_a2a<commtype>(coll_handle_, stream, instance_id_);
  HCTR_LIB_THROW(cudaEventRecord(comm_event_, comm_stream_));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, comm_event_));
  // ib_comm_->wait_global_recv_async(coll_handle_, instance_id_);
}

template <typename commtype>
void HierAll2Allv_Multi_IB<commtype>::initiate_communication(cudaStream_t stream) {
  ib_comm_->post_a2a_send_command<commtype>(coll_handle_, stream, instance_id_);
  HCTR_LIB_THROW(cudaEventRecord(comm_event_, comm_stream_));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, comm_event_));
}

template <typename commtype>
void HierAll2Allv_Multi_IB<commtype>::wait_completion(cudaStream_t stream) {
  ib_comm_->blocking_wait(coll_handle_, stream, instance_id_);
  HCTR_LIB_THROW(cudaEventRecord(comm_event_, comm_stream_));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream, comm_event_));
  // ib_comm_->wait_global_recv_async(coll_handle_, instance_id_);
}

template <typename commtype>
HierAll2Allv_Multi_IB<commtype>::~HierAll2Allv_Multi_IB() {
  cudaEventDestroy(comm_event_);
}
#endif

template class AllToAllVComm<float>;
template class AllToAllVComm<__half>;
template class AllReduceComm<float>;
template class AllReduceComm<__half>;

template class AllToAll_Multi_NCCL<float>;
template class AllToAll_Multi_NCCL<__half>;
#ifdef ENABLE_MPI
template class HierAll2Allv_Multi_IB<float>;
template class HierAll2Allv_Multi_IB<__half>;
#endif

}  // namespace hybrid_embedding

}  // namespace HugeCTR
