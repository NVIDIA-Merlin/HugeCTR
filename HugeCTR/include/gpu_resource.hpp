/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 private:
  cudaStream_t stream_;           /**< cuda stream for computation */
  cudaStream_t data_copy_stream_; /**< cuda stream for data copy */
  cublasHandle_t cublas_handle_;
  cudnnHandle_t cudnn_handle_;
  const int device_id_;
  const ncclComm_t* comm_;

 public:
  /**
   * Ctor
   */
  GPUResource(int device_id, const ncclComm_t* comm) : device_id_(device_id), comm_(comm) {
    int o_device = -1;
    CK_CUDA_THROW_(get_set_device(device_id_, &o_device));
    CK_CUBLAS_THROW_(cublasCreate(&cublas_handle_));
    CK_CUDNN_THROW_(cudnnCreate(&cudnn_handle_));
    CK_CUDA_THROW_(cudaStreamCreate(&stream_));
    CK_CUDA_THROW_(cudaStreamCreate(&data_copy_stream_));
    CK_CUDA_THROW_(get_set_device(o_device));
    return;
  }

  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;

  /*
   * Dtor
   */
  ~GPUResource() {
    try {
      int o_device = -1;
      CK_CUDA_THROW_(get_set_device(device_id_, &o_device));
      CK_CUBLAS_THROW_(cublasDestroy(cublas_handle_));
      CK_CUDNN_THROW_(cudnnDestroy(cudnn_handle_));
      CK_CUDA_THROW_(cudaStreamDestroy(stream_));
      CK_CUDA_THROW_(cudaStreamDestroy(data_copy_stream_));
      CK_CUDA_THROW_(get_set_device(o_device));
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
    return;
  }
  int get_device_id() const { return device_id_; }
  const cudaStream_t* get_stream_ptr() const { return &stream_; }
  const cudaStream_t* get_data_copy_stream_ptr() const { return &data_copy_stream_; }
  const cublasHandle_t* get_cublas_handle_ptr() const { return &cublas_handle_; }
  const cudnnHandle_t* get_cudnn_handle_ptr() const { return &cudnn_handle_; }
  const ncclComm_t* get_nccl_ptr() const { return comm_; }
};

/**
 * @brief GPU resources container.
 *
 * A GPU resource container in one node. An instant includes:
 * GPU resource vector, thread pool for training, nccl communicators.
 */
class GPUResourceGroup {
 private:
  ncclComm_t* comms_;
  const DeviceMap device_map_;
  std::vector<GPUResource*> gpu_resources_; /**< GPU resource vector */
 public:
  ctpl::thread_pool train_thread_pool; /**< cpu thread pool for training */
  std::vector<std::future<void>> results;

  GPUResourceGroup(const DeviceMap& device_map)
      : comms_(nullptr),
        device_map_(device_map),
        gpu_resources_(device_map_.get_device_list().size(), nullptr),
        train_thread_pool(device_map_.get_device_list().size()),
        results(device_map_.get_device_list().size()) {
    auto& device_list = device_map_.get_device_list();
    size_t local_gpu_count = device_list.size();

    if (local_gpu_count == 0) {
      CK_THROW_(Error_t::WrongInput, "Empty device_list");
    }
    if (local_gpu_count != size()) {
      CK_THROW_(Error_t::WrongInput, "local_gpu_count != size()");
    }
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    for (int dev : device_list) {
      if (dev >= dev_count) {
        CK_THROW_(Error_t::WrongInput, "Invalid device id: " + std::to_string(dev));
      }
    }

    if (gpu_resources_.size() != local_gpu_count) {
      CK_THROW_(Error_t::WrongInput, "gpu_resource_.size() != local_gpu_count");
    }
    int total_gpu_count = get_total_gpu_count();
    // if ther are multiple GPUs within a node or/and across nodes
    if (total_gpu_count > 1) {
      int my_rank = 0;
      int n_ranks = 1;
      comms_ = new ncclComm_t[local_gpu_count]();
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
      CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
      ncclUniqueId nid;
      if (my_rank == 0) CK_NCCL_THROW_(ncclGetUniqueId(&nid));
      CK_MPI_THROW_(MPI_Bcast((void*)&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD));

      CK_NCCL_THROW_(ncclGroupStart());
      for (size_t i = 0; i < local_gpu_count; i++) {
        CK_CUDA_THROW_(cudaSetDevice(device_list[i]));
        CK_NCCL_THROW_(ncclCommInitRank(comms_ + i, total_gpu_count, nid,
                                        device_map_.get_global_id(device_list[i])));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
#else
      CK_NCCL_THROW_(ncclCommInitAll(comms_, device_list.size(), device_list.data()));
#endif
    } else {
      comms_ = nullptr;
    }
    for (size_t i = 0; i < local_gpu_count; i++) {
      gpu_resources_[i] = (new GPUResource(device_list[i], comms_ + i));
    }
  }

  GPUResourceGroup(const GPUResourceGroup& C) = delete;
  GPUResourceGroup& operator=(const GPUResourceGroup&) = delete;

  const GPUResource* operator[](int idx) const { return gpu_resources_[idx]; }
  size_t size() const {
    // return gpu_resources_.size();
    return device_map_.get_device_list().size();
  }
  bool empty() const { return size() == 0; }
  ~GPUResourceGroup() {
    try {
      if (gpu_resources_.size() > 1) {
        for (unsigned int i = 0; i < gpu_resources_.size(); i++) {
          CK_NCCL_THROW_(ncclCommDestroy(comms_[i]));
        }
        delete[] comms_;
      }
      for (auto gpu_resource : gpu_resources_) {
        delete gpu_resource;
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }

  const std::vector<int>& get_device_list() const { return device_map_.get_device_list(); }
  int get_global_id(int local_device_id) const {
    return device_map_.get_global_id(local_device_id);
  }
  int get_local_id(int global_id) const {  // sequential GPU indices
    return device_map_.get_local_id(global_id);
  }
  int get_local_device_id(int global_id) const {  // the actual GPU ids
    return device_map_.get_local_device_id(global_id);
  }
  int get_total_gpu_count() const { return device_map_.size(); }
  int get_node_count() const { return device_map_.num_nodes(); }
  int get_pid(int global_id) const { return device_map_.get_pid(global_id); }
};

}  // namespace HugeCTR
