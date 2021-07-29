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

#pragma once

#include <cuda_runtime.h>

#include <collectives/all_reduce_comm.hpp>
#include <collectives/ib_comm.hpp>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

class Communication {
 public:
  Communication(size_t width_data_field);
  virtual ~Communication() = default;
  virtual void communicate(cudaStream_t stream) = 0;
  virtual void update_sizes(cudaStream_t stream){};
  virtual void initiate_communication(cudaStream_t stream){};
  virtual void wait_completion(cudaStream_t stream){};

 protected:
  size_t width_data_field_;
};

/*
 * All to All communications
 */
template <typename commtype>
struct AllToAllStorage {
  AllToAllStorage(GeneralBuffer2<CudaAllocator>* buf, size_t max_buffer_size) {
    buf->reserve({max_buffer_size}, &send_buffer);
    buf->reserve({max_buffer_size}, &recv_buffer);
  }
  Tensor2<commtype> send_buffer, recv_buffer;
  Tensor2<commtype*> send_buffer_ptrs;
};

template <typename commtype>
class AllToAllVComm : public Communication {
 public:
  AllToAllVComm(Tensor2<commtype> send_buffer, Tensor2<commtype> recv_buffer,
                const uint32_t* send_offsets, const uint32_t* recv_offsets,
                const GPUResource* gpu_resource, size_t width_data_field);

 protected:
  Tensor2<commtype> send_buffer_;
  Tensor2<commtype> recv_buffer_;

  const uint32_t* send_offsets_;
  const uint32_t* recv_offsets_;

  const GPUResource* gpu_resource_;
};

template <typename commtype>
class AllToAll_Multi_NCCL : public AllToAllVComm<commtype> {
 public:
  using AllToAllVComm<commtype>::AllToAllVComm;
  void communicate(cudaStream_t stream) final override;
  ~AllToAll_Multi_NCCL() = default;
};

// template <typename commtype>
// class AllToAll_Single : public AllToAllVComm<commtype> {
// public:
//   using AllToAllVComm<commtype>::AllToAllVComm;
//   void communicate() final override;
//   ~AllToAll_Single() = default;
// };

/*
 * All Reduce communications
 */
template <typename commtype>
class AllReduceComm : public Communication {
 public:
  AllReduceComm(AllReduceInPlaceComm* ar_comm, AllReduceInPlaceComm::Handle ar_handle,
                const GPUResource* gpu_resource);
  void communicate(cudaStream_t stream) final override;
  ~AllReduceComm() = default;

 private:
  AllReduceInPlaceComm* ar_comm_;
  AllReduceInPlaceComm::Handle ar_handle_;
  const GPUResource* gpu_resource_;
};

#ifdef ENABLE_MPI
template <typename commtype>
class HierAll2Allv_Multi_IB : public Communication {
 public:
  HierAll2Allv_Multi_IB(uint32_t instance_id, HierA2AvCollHandle coll_handle, size_t** send_sizes,
                        const GPUResource* gpu_resource, IbComm* ib_comm, cudaStream_t comm_stream);

  void update_sizes(cudaStream_t stream) final override;
  void communicate(cudaStream_t stream) final override;
  void initiate_communication(cudaStream_t stream) final override;
  void wait_completion(cudaStream_t stream) final override;
  ~HierAll2Allv_Multi_IB();

 private:
  uint32_t instance_id_;
  HierA2AvCollHandle coll_handle_;
  size_t** send_sizes_;
  const GPUResource* gpu_resource_;
  IbComm* ib_comm_;
  cudaStream_t comm_stream_;
  cudaEvent_t comm_event_;
};
#endif

}  // namespace hybrid_embedding

}  // namespace HugeCTR
