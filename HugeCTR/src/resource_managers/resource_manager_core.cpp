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

#include <omp.h>

#include <random>
#include <resource_managers/resource_manager_core.hpp>
#include <utils.hpp>

namespace HugeCTR {

void ResourceManagerCore::all2all_warmup() {
  auto num_global_gpus = get_global_gpu_count();
  auto num_local_gpus = get_local_gpu_count();
  Tensors2<uint64_t> all2all_warmup_tensors(num_local_gpus);

  // Allocate temp buffers
  for (size_t g = 0; g < get_local_gpu_count(); g++) {
    auto& local_gpu = get_local_gpu(g);
    CudaDeviceContext context(local_gpu->get_device_id());
    auto buf = GeneralBuffer2<CudaAllocator>::create();
    buf->reserve({num_global_gpus}, &all2all_warmup_tensors[g]);
    buf->allocate();
  }

  // Do all2all warmup
  MESSAGE_("Start all2all warmup");
  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t g = 0; g < get_local_gpu_count(); g++) {
    auto& local_gpu = get_local_gpu(g);
    CudaDeviceContext context(local_gpu->get_device_id());
    auto& stream = local_gpu->get_stream();
    auto& comm = local_gpu->get_nccl();
    for (size_t s = 0; s < num_global_gpus; s++) {
      CK_NCCL_THROW_(ncclSend(all2all_warmup_tensors[g].get_ptr() + s, 1, ncclUint64, s, comm, stream));
      CK_NCCL_THROW_(ncclRecv(all2all_warmup_tensors[g].get_ptr() + s, 1, ncclUint64, s, comm, stream));
    }
  }
  CK_NCCL_THROW_(ncclGroupEnd());
  MESSAGE_("End all2all warmup");
}

ResourceManagerCore::ResourceManagerCore(int num_process, int process_id, DeviceMap&& device_map,
                                 unsigned long long seed)
    : num_process_(num_process), process_id_(process_id), device_map_(std::move(device_map)) {
  // set threads affinity

  if (num_process_ != device_map_.num_nodes()) {
    CK_THROW_(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
  }

  auto& local_gpu_device_id_list = device_map_.get_device_list();
  size_t local_gpu_count = local_gpu_device_id_list.size();
  size_t global_gpu_count = device_map_.size();

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

  std::mt19937 gen(seed);
  std::uniform_int_distribution<unsigned long long> dis;

  unsigned long long replica_uniform_seed = dis(gen);
  std::vector<unsigned long long> replica_variant_seeds(global_gpu_count);
  for (size_t i = 0; i < replica_variant_seeds.size(); i++) {
    replica_variant_seeds[i] = dis(gen);
  }

  std::vector<unsigned long long> local_replica_variant_seeds(local_gpu_count);
  for (size_t i = 0; i < local_replica_variant_seeds.size(); i++) {
    local_replica_variant_seeds[i] = replica_variant_seeds[device_map_.get_global_id(i)];
  }

  cpu_resource_.reset(new CPUResource(replica_uniform_seed, local_replica_variant_seeds));

  CudaDeviceContext context;
  std::vector<ncclComm_t> comms(local_gpu_count);
#ifdef ENABLE_MPI
  ncclUniqueId nid;
  if (process_id_ == 0) CK_NCCL_THROW_(ncclGetUniqueId(&nid));
  CK_MPI_THROW_(MPI_Bcast((void*)&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD));

  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    context.set_device(local_gpu_device_id_list[i]);
    CK_NCCL_THROW_(
        ncclCommInitRank(&comms[i], global_gpu_count, nid, device_map_.get_global_id(i)));
  }
  CK_NCCL_THROW_(ncclGroupEnd());
#else
  CK_NCCL_THROW_(ncclCommInitAll(comms.data(), local_gpu_device_id_list.size(),
                                 local_gpu_device_id_list.data()));
#endif

  gpu_resources_.resize(local_gpu_count);
#pragma omp parallel num_threads(local_gpu_count)
  {
    size_t id = omp_get_thread_num();
    set_local_gpu(std::make_shared<GPUResource>(local_gpu_device_id_list[id],
                                                id,
                                                device_map_.get_global_id(id),
                                                replica_uniform_seed,
                                                local_replica_variant_seeds[id],
                                                comms[id]),
                  id);
  }

  for (size_t i = 0; i < local_gpu_count; i++) {
    p2p_matrix_.push_back(std::vector<bool>(local_gpu_count, false));
  }

  enable_all_peer_accesses();
  if (all_p2p_enabled() == false) {
    MESSAGE_("Peer-to-peer access cannot be fully enabled.");
  }

  all2all_warmup();

  if (num_process_ > 1) {
#ifdef ENABLE_MPI
    ib_comm_ = std::make_unique<IbComm>();
    ib_comm_->init(num_process_, local_gpu_count, process_id_, local_gpu_device_id_list);
#endif
  }
}

bool ResourceManagerCore::p2p_enabled(int src_device_id, int dst_device_id) const {
  return p2p_matrix_[src_device_id][dst_device_id];
}

bool ResourceManagerCore::all_p2p_enabled() const {
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

void ResourceManagerCore::enable_all_peer_accesses() {
  const auto& local_gpu_device_id_list = device_map_.get_device_list();
  size_t local_gpu_count = local_gpu_device_id_list.size();

  assert(local_gpu_count != 0);

#pragma omp parallel num_threads(local_gpu_count)
  {
    size_t id = omp_get_thread_num();
    CudaDeviceContext context(local_gpu_device_id_list[id]);
    for (size_t j = 0; j < local_gpu_count; j++) {
      if (id != j) {
        int can_access_peer;
        CK_CUDA_THROW_(cudaDeviceCanAccessPeer(&can_access_peer, local_gpu_device_id_list[id],
                                               local_gpu_device_id_list[j]));
        if (can_access_peer == 1) {
          p2p_matrix_[id][j] = true;
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

void ResourceManagerCore::set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision)
{
#ifdef ENABLE_MPI
  ar_comm_ = AllReduceInPlaceComm::create(num_process_, algo, use_mixed_precision, gpu_resources_,
                                          ib_comm_.get());
#else
  ar_comm_ = AllReduceInPlaceComm::create(num_process_, algo, use_mixed_precision, gpu_resources_);
#endif
}


}  // namespace HugeCTR
