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

#include "HugeCTR/include/gpu_resource.hpp"

namespace HugeCTR {

GPUResource::GPUResource(int device_id, const ncclComm_t* comm)
    : device_id_(device_id), comm_(comm) {
  CudaDeviceContext context(device_id_);
  CK_CUBLAS_THROW_(cublasCreate(&cublas_handle_));
  CK_CURAND_THROW_(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CK_CUDNN_THROW_(cudnnCreate(&cudnn_handle_));
  CK_CUDA_THROW_(cudaStreamCreate(&stream_));
  CK_CUDA_THROW_(cudaEventCreate(&event_));
  CK_CUDA_THROW_(cudaStreamCreate(&data_copy_stream_[0]));
  CK_CUDA_THROW_(cudaStreamCreate(&data_copy_stream_[1]));
}

GPUResource::~GPUResource() {
  try {
    CudaDeviceContext context(device_id_);
    CK_CUBLAS_THROW_(cublasDestroy(cublas_handle_));
    CK_CURAND_THROW_(curandDestroyGenerator(curand_generator_));
    CK_CUDNN_THROW_(cudnnDestroy(cudnn_handle_));
    CK_CUDA_THROW_(cudaStreamDestroy(stream_));
    CK_CUDA_THROW_(cudaEventDestroy(event_));
    CK_CUDA_THROW_(cudaStreamDestroy(data_copy_stream_[0]));
    CK_CUDA_THROW_(cudaStreamDestroy(data_copy_stream_[1]));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

GPUResourceGroup::GPUResourceGroup(const std::shared_ptr<const DeviceMap>& device_map)
    : comms_(nullptr),
      device_map_(device_map),
      train_thread_pool(device_map->get_device_list().size()),
      results(device_map->get_device_list().size()) {
  // set threads affinity
  for (unsigned int i = 0; i < device_map->get_device_list().size(); i++) {
    set_affinity(train_thread_pool.get_thread(i), {}, true);
  }

  auto& device_list = device_map->get_device_list();
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

  if (device_map->get_device_list().size() != local_gpu_count) {
    CK_THROW_(Error_t::WrongInput,
              "device_map->get_device_list().size() != local_gpu_count");
  }
  int total_gpu_count = get_total_gpu_count();
  // if ther are multiple GPUs within a node or/and across nodes
  if (total_gpu_count > 1) {
  comms_.reset(new ncclComm_t[local_gpu_count]());
#ifdef ENABLE_MPI
  int my_rank = 0;
  int n_ranks = 1;
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
  ncclUniqueId nid;
  if (my_rank == 0) CK_NCCL_THROW_(ncclGetUniqueId(&nid));
  CK_MPI_THROW_(MPI_Bcast((void*)&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD));

  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    CK_CUDA_THROW_(cudaSetDevice(device_list[i]));
    CK_NCCL_THROW_(ncclCommInitRank(comms_.get() + i, total_gpu_count, nid,
                                    device_map->get_global_id(device_list[i])));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
#else
  CK_NCCL_THROW_(ncclCommInitAll(comms_.get(), device_list.size(), device_list.data()));
#endif
  }
  for (size_t i = 0; i < local_gpu_count; i++) {
    gpu_resources_.emplace_back(new GPUResource(device_list[i], comms_.get() + i));
  }

  enable_all_peer_accesses();
  if (all_p2p_enabled() == false) {
    MESSAGE_("Peer-to-peer access cannot be fully enabled.");
  }
}

GPUResourceGroup::~GPUResourceGroup() {
  try {
    if (gpu_resources_.size() > 1) {
      for (unsigned int i = 0; i < gpu_resources_.size(); i++) {
        CK_NCCL_THROW_(ncclCommDestroy(comms_[i]));
      }
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

bool GPUResourceGroup::all_p2p_enabled() const {
  for (int src_dev = 0; src_dev < (int)p2p_enabled_.size(); src_dev++) {
    for (int dst_dev = 0; dst_dev < (int)p2p_enabled_[src_dev].size(); dst_dev++) {
      bool enabled = (dst_dev == src_dev) || p2p_enabled_[src_dev][dst_dev];
      if (!enabled) {
        return false;
      }
    }
  }
  return true;
}

void GPUResourceGroup::enable_all_peer_accesses() {
  assert(!empty());

  int num_gpus = get_local_gpu_count();

  p2p_enabled_.resize(num_gpus);
  for (int dev = 0; dev < num_gpus; dev++) {
    p2p_enabled_[dev].resize(num_gpus, false);
  }

  for (int src_dev = 0; src_dev < num_gpus; src_dev++) {
    CudaDeviceContext context(src_dev);
    for (int dst_dev = 0; dst_dev < num_gpus; dst_dev++) {
      if (dst_dev != src_dev) {
        int can_access_peer = 0;
        CK_CUDA_THROW_(cudaDeviceCanAccessPeer(&can_access_peer, src_dev, dst_dev));
        if (can_access_peer) {
          cudaError_t ret = cudaDeviceEnablePeerAccess(dst_dev, 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            CK_CUDA_THROW_(ret);
          } else {
            // cudaErrorPeerAccessAlreadyEnabled must not be handled as an error
            // so we reset it to cudaSuccess here
            cudaGetLastError();
          }
        }
        p2p_enabled_[src_dev][dst_dev] = (bool)can_access_peer;
      }
    }
  }
}

}  // namespace HugeCTR
