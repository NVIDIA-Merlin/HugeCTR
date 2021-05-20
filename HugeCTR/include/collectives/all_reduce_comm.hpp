/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <boost/serialization/strong_typedef.hpp>
#include <collectives/ib_comm.hpp>
#include <collectives/ib_proxy.hpp>
#include <gpu_resource.hpp>
#include <memory>
#include <vector>
#include <tensor2.hpp>
#include <general_buffer2.hpp>

namespace HugeCTR {
enum class AllReduceAlgo { ONESHOT, NCCL };

class AllReduceInPlaceComm {
 public:
  BOOST_STRONG_TYPEDEF(size_t, Handle)

  virtual Handle register_coll() = 0;
  virtual void set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t device_id) = 0;
  virtual void register_coll_buf(Handle coll) = 0;
  virtual void update_size(Handle coll, const size_t ar_size) = 0;
  virtual void all_reduce(Handle coll, cudaStream_t stream, size_t device_id) = 0;
#ifdef ENABLE_MPI
  static std::shared_ptr<AllReduceInPlaceComm> create(
      size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm);
#else
  static std::shared_ptr<AllReduceInPlaceComm> create(
      size_t num_process, AllReduceAlgo algo, bool use_mixed_precision,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);
#endif
 private:
#ifdef ENABLE_MPI
  static std::shared_ptr<AllReduceInPlaceComm> create_oneshot(
      size_t num_process, bool use_mixed_precision,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources, IbComm* ib_comm);
#endif
  static std::shared_ptr<AllReduceInPlaceComm> create_nccl(
      size_t num_process, bool use_mixed_precision,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);
  static std::shared_ptr<AllReduceInPlaceComm> create_oneshot(
      size_t num_process, bool use_mixed_precision,
      const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);
};

#ifdef ENABLE_MPI
template <typename T>
class OneshotMultiARInplaceComm : public AllReduceInPlaceComm {
 public:
  virtual Handle register_coll() final;
  virtual void set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t device_id) final;
  virtual void register_coll_buf(Handle coll) final;
  virtual void update_size(Handle coll, const size_t ar_size) final;
  virtual void all_reduce(Handle coll, cudaStream_t stream, size_t device_id) final;

  OneshotMultiARInplaceComm(IbComm* ib_comm, size_t num_procs,
                            const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);

 private:
  struct ARContextPerGPU {
    void* ar_ptr_ = NULL;
    cudaStream_t stream;
  };

  struct ARContext {
    std::vector<ARContextPerGPU> ctx_;
    size_t ar_size_ = 0;
    ARCollHandle ib_comm_handle_;
  };

  IbComm* ib_comm_;
  const std::vector<std::shared_ptr<GPUResource>>& gpu_resources_;
  std::vector<std::unique_ptr<ARContext>> ar_ctx_;
  size_t num_procs_ = 1;
  size_t num_gpus_ = 1;
};
#endif
template <typename T>
class OneshotSingleARInplaceComm : public AllReduceInPlaceComm {
 public:
  virtual Handle register_coll() final;
  virtual void set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t device_id) final;
  virtual void register_coll_buf(Handle coll) final;
  virtual void update_size(Handle coll, const size_t ar_size) final;
  virtual void all_reduce(Handle coll, cudaStream_t stream, size_t device_id) final;

  OneshotSingleARInplaceComm(const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);

 private:
  struct ARContextPerGPU {
    void* ar_ptr_ = NULL;
    cudaStream_t stream;
    Tensor2<void*> d_peer_ptrs_;
    Tensor2<size_t> d_coll_cmd_;
    Tensor2<size_t> d_flags_;
    Tensor2<size_t*> d_flags_ptrs_;
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf_ = NULL;
  };

  struct ARContext {
    std::vector<ARContextPerGPU> ctx_;
    size_t ar_size_ = 0;
  };

  const std::vector<std::shared_ptr<GPUResource>>& gpu_resources_;
  std::vector<std::unique_ptr<ARContext>> ar_ctx_;
  size_t num_gpus_ = 1;
  std::vector<int> device_list_;
  int cfg_nchannels_ = 16;
};

template <typename T>
class NCCLARInplaceComm : public AllReduceInPlaceComm {
 public:
  virtual Handle register_coll() final;
  virtual void set_coll_buf(Handle coll, void* ar_ptr, size_t ar_size, size_t device_id) final;
  virtual void register_coll_buf(Handle coll) final;
  virtual void update_size(Handle coll, const size_t ar_size) final;
  virtual void all_reduce(Handle coll, cudaStream_t stream, size_t device_id) final;

  NCCLARInplaceComm(size_t num_procs,
                    const std::vector<std::shared_ptr<GPUResource>>& gpu_resources);

 private:
  struct ARContextPerGPU {
    void* ar_ptr_ = NULL;
    cudaStream_t stream;
  };

  struct ARContext {
    std::vector<ARContextPerGPU> ctx_;
    size_t ar_size_ = 0;
  };

  const std::vector<std::shared_ptr<GPUResource>>& gpu_resources_;
  std::vector<std::unique_ptr<ARContext>> ar_ctx_;
  size_t num_procs_ = 1;
  size_t num_gpus_ = 1;
};
}  // namespace HugeCTR
