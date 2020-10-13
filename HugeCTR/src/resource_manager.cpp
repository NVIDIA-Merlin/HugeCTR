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

#include <random>
#include <resource_manager.hpp>
#include <utils.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#pragma GCC diagnostic pop

namespace HugeCTR {

std::shared_ptr<ResourceManager> ResourceManager::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed) {
  int size = 1, rank = 0;
#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &size));
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
#endif

  DeviceMap device_map(visible_devices, rank);

  std::random_device rd;
  if (seed == 0) {
    seed = rd();
  }
  MESSAGE_("Initial seed is " + std::to_string(seed));

  return std::shared_ptr<ResourceManager>(
      new ResourceManager(size, rank, std::move(device_map), seed));
}

ResourceManager::ResourceManager(int num_process, int pid, DeviceMap&& device_map,
                                 unsigned long long seed)
    : num_process_(num_process), pid_(pid), device_map_(std::move(device_map)) {
  // set threads affinity

  if (num_process_ != device_map_.num_nodes()) {
    CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
  }

  std::mt19937 gen(seed);
  std::uniform_int_distribution<unsigned long long> dis;

  auto& local_gpu_device_id_list = device_map_.get_device_list();
  size_t local_gpu_count = local_gpu_device_id_list.size();

  if (local_gpu_count == 0) {
    CK_THROW_(Error_t::WrongInput, "No local gpu");
  }

  // Question, what if user use CUDA_VISIBLE_DEVICES
  int dev_count;
  CK_CUDA_THROW_(cudaGetDeviceCount(&dev_count));
  for (int device_id : local_gpu_device_id_list) {
    if (device_id >= dev_count) {
      CK_THROW_(Error_t::WrongInput, "Invalid device id: " + std::to_string(device_id));
    }
  }

  int total_gpu_count = device_map_.size();
  // if ther are multiple GPUs within a node or/and across nodes
  if (total_gpu_count > 1) {
    std::vector<ncclComm_t> comms(local_gpu_count);
#ifdef ENABLE_MPI
    ncclUniqueId nid;
    if (pid_ == 0) CK_NCCL_THROW_(ncclGetUniqueId(&nid));
    CK_MPI_THROW_(MPI_Bcast((void*)&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD));

    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t i = 0; i < local_gpu_count; i++) {
      CK_CUDA_THROW_(cudaSetDevice(local_gpu_device_id_list[i]));
      CK_NCCL_THROW_(ncclCommInitRank(&comms[i], total_gpu_count, nid,
                                      device_map_.get_global_id(i)));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
#else
    CK_NCCL_THROW_(ncclCommInitAll(comms.data(), local_gpu_device_id_list.size(),
                                   local_gpu_device_id_list.data()));
#endif
    for (size_t i = 0; i < local_gpu_count; i++) {
      gpu_resources_.emplace_back(new GPUResource(
          local_gpu_device_id_list[i], device_map_.get_global_id(i),
          dis(gen), comms[i]));
    }
  } else {
    gpu_resources_.emplace_back(
        new GPUResource(local_gpu_device_id_list[0],
                        device_map_.get_global_id(0), dis(gen)));
  }

  cpu_resource_.reset(new CPUResource(dis(gen), device_map_.get_device_list().size()));

  for (size_t i = 0; i < local_gpu_count; i++) {
    p2p_matrix_.push_back(std::vector<bool>(local_gpu_count, false));
  }

  enable_all_peer_accesses();
  if (all_p2p_enabled() == false) {
    MESSAGE_("Peer-to-peer access cannot be fully enabled.");
  }

  const size_t pool_alloc_size = 256 * 1024 * 1024;
  std::vector<int> device_id_list = device_map_.get_device_list();
  memory_resource_ =
      std::make_shared<rmm::mr::cnmem_memory_resource>(pool_alloc_size, device_id_list);
}

bool ResourceManager::p2p_enabled(int src_device_id, int dst_device_id) const {
  return p2p_matrix_[src_device_id][dst_device_id];
}

bool ResourceManager::all_p2p_enabled() const {
  size_t num_gpus = get_local_gpu_count();
  if (num_gpus == 1) {
    return false;
  }

  for (size_t i = 0; i < num_gpus; i++) {
    for (size_t j = 0; j < num_gpus; j++) {
      if (i != j && !p2p_matrix_[i][j]) return false;
    }
  }

  return true;
}

void ResourceManager::enable_all_peer_accesses() {
  const auto& local_gpu_device_id_list = device_map_.get_device_list();
  size_t local_gpu_count = local_gpu_device_id_list.size();

  assert(local_gpu_count != 0);

  for (size_t i = 0; i < local_gpu_count; i++) {
    CudaDeviceContext context(local_gpu_device_id_list[i]);
    for (size_t j = 0; j < local_gpu_count; j++) {
      if (i != j) {
        int can_access_peer;
        CK_CUDA_THROW_(cudaDeviceCanAccessPeer(&can_access_peer, local_gpu_device_id_list[i],
                                               local_gpu_device_id_list[j]));
        if (can_access_peer == 1) {
          p2p_matrix_[i][j] = true;
          cudaError_t ret = cudaDeviceEnablePeerAccess(local_gpu_device_id_list[j], 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            CK_CUDA_THROW_(ret);
          } else {
            // cudaErrorPeerAccessAlreadyEnabled must not be handled as an error
            // so we reset it to cudaSuccess here
            cudaGetLastError();
          }
        }
      }
    }
  }
}
}  // namespace HugeCTR
