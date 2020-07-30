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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/device_map.hpp"
#include "HugeCTR/include/utils.hpp"
#include "ctpl/ctpl_stl.h"

#include <cudnn.h>
#include <curand.h>
#include <nccl.h>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

/**
 * @brief GPU resource allocated on a target gpu.
 *
 * This class implement unified resource managment on the target GPU.
 */
class GPUResource {
 public:
  GPUResource(int device_id);
  GPUResource(int device_id, const ncclComm_t& comm);
  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;
  ~GPUResource();

  int get_device_id() const { return device_id_; }
  const cudaStream_t& get_stream() const { return stream_; }
  const cudaStream_t& get_data_copy_stream(int id) const { return data_copy_stream_[0]; }
  const cublasHandle_t& get_cublas_handle() const { return cublas_handle_; }
  const curandGenerator_t& get_curand_generator() const { return curand_generator_; }
  const cudnnHandle_t& get_cudnn_handle() const { return cudnn_handle_; }
  const ncclComm_t& get_nccl() const { return comm_; }
  const cudaEvent_t& get_event() const { return event_; }

  bool support_NCCL() const { return comm_ != nullptr; }

 private:
  const int device_id_;
  cudaStream_t stream_;              /**< cuda stream for computation */
  cudaStream_t data_copy_stream_[2]; /**< cuda stream for data copy */
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  cudnnHandle_t cudnn_handle_;
  cudaEvent_t event_;
  ncclComm_t comm_;
};

/**
 * @brief GPU resources container.
 *
 * A GPU resource container in one node. An instant includes:
 * GPU resource vector, thread pool for training, nccl communicators.
 */
class GPUResourceGroup {
 public:
  GPUResourceGroup(const std::shared_ptr<const DeviceMap>& device_map);
  GPUResourceGroup(const GPUResourceGroup&) = delete;
  GPUResourceGroup& operator=(const GPUResourceGroup&) = delete;
  ~GPUResourceGroup();

  const GPUResource& operator[](int idx) const { return *gpu_resources_[idx]; }
  const std::shared_ptr<const GPUResource>& get_shared(int idx) { return gpu_resources_[idx]; }
  size_t size() const { return device_map_->get_device_list().size(); }
  bool empty() const { return size() == 0; }

  const std::vector<int>& get_device_list() const { return device_map_->get_device_list(); }
  int get_global_id(int local_device_id) const {
    return device_map_->get_global_id(local_device_id);
  }
  int get_local_id(int global_id) const {  // sequential GPU indices
    return device_map_->get_local_id(global_id);
  }
  int get_local_device_id(int global_id) const {  // the actual GPU ids
    return device_map_->get_local_device_id(global_id);
  }
  int get_local_gpu_count() const { return get_device_list().size(); }
  int get_total_gpu_count() const { return device_map_->size(); }
  int get_node_count() const { return device_map_->num_nodes(); }
  int get_pid(int global_id) const { return device_map_->get_pid(global_id); }

  bool p2p_enabled(int src_dev, int dst_dev) const;
  bool all_p2p_enabled() const;

  ctpl::thread_pool& get_thread_pool() { return thread_pool_; }

 private:
  void enable_all_peer_accesses();

  std::shared_ptr<const DeviceMap> device_map_;
  std::vector<std::shared_ptr<const GPUResource>> gpu_resources_; /**< GPU resource vector */
  std::map<int, std::map<int, bool>> p2p_enabled_;

  ctpl::thread_pool thread_pool_; /**< cpu thread pool for training */
};

using GPUResourceGroupPtr = std::shared_ptr<GPUResourceGroup>;

}  // namespace HugeCTR
