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

#include <omp.h>

#include <core23/logger.hpp>
#include <cstdlib>
#include <random>
#include <resource_managers/resource_manager_core.hpp>
#include <utils.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#pragma GCC diagnostic pop

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

#ifndef DISABLE_A2A_WARMUP
  // Do all2all warmup
  HCTR_LOG_S(INFO, ROOT) << "Start all2all warmup" << std::endl;
  HCTR_LIB_THROW(ncclGroupStart());
  for (size_t g = 0; g < get_local_gpu_count(); g++) {
    auto& local_gpu = get_local_gpu(g);
    CudaDeviceContext context(local_gpu->get_device_id());
    auto& stream = local_gpu->get_stream();
    auto& comm = local_gpu->get_nccl();
    for (size_t s = 0; s < num_global_gpus; s++) {
      HCTR_LIB_THROW(
          ncclSend(all2all_warmup_tensors[g].get_ptr() + s, 1, ncclUint64, s, comm, stream));
      HCTR_LIB_THROW(
          ncclRecv(all2all_warmup_tensors[g].get_ptr() + s, 1, ncclUint64, s, comm, stream));
    }
  }
  HCTR_LIB_THROW(ncclGroupEnd());
  HCTR_LOG_S(INFO, ROOT) << "End all2all warmup" << std::endl;
#else
  HCTR_LOG_S(INFO, ROOT) << "Skip all2all warmup" << std::endl;
#endif
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
        HCTR_LIB_THROW(cudaDeviceCanAccessPeer(&can_access_peer, local_gpu_device_id_list[id],
                                               local_gpu_device_id_list[j]));
        if (can_access_peer == 1) {
          p2p_matrix_[id][j] = true;
          cudaError_t ret = cudaDeviceEnablePeerAccess(local_gpu_device_id_list[j], 0);
          if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
            HCTR_LIB_THROW(ret);
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

void ResourceManagerCore::initialize_rmm_resources() {
  const size_t pool_alloc_size = 256 * 1024 * 1024;
  using dmmr = rmm::mr::device_memory_resource;
  static const char* allow_set_char = getenv("HCTR_RMM_SETTABLE");
  bool allow_set = true;
  if (allow_set_char && allow_set_char[0] == '0') {
    allow_set = false;
  }
  CudaDeviceContext context;
  auto local_gpu_device_id_list = get_local_gpu_device_id_list();
  for (size_t i = 0; i < local_gpu_device_id_list.size(); i++) {
    context.set_device(local_gpu_device_id_list[i]);
    base_cuda_mr_.emplace_back(std::make_shared<rmm::mr::cuda_memory_resource>());
    memory_resource_.emplace_back(std::make_shared<rmm::mr::pool_memory_resource<dmmr>>(
        base_cuda_mr_.back().get(), pool_alloc_size));
    if (allow_set) {
      original_device_resource_.push_back(
          rmm::mr::set_current_device_resource(memory_resource_.back().get()));
    }
  }
}
ResourceManagerCore::ResourceManagerCore(int num_process, int process_id, DeviceMap&& device_map,
                                         unsigned long long seed)
    : num_process_(num_process), process_id_(process_id), device_map_(std::move(device_map)) {
  if (num_process_ != device_map_.num_nodes()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: the MPI total rank doesn't match the node count");
  }

  auto& local_gpu_device_id_list = device_map_.get_device_list();
  size_t local_gpu_count = local_gpu_device_id_list.size();
  size_t global_gpu_count = device_map_.size();

  if (local_gpu_count == 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "No local gpu");
  }

  // Question, what if user use CUDA_VISIBLE_DEVICES
  int dev_count;
  HCTR_LIB_THROW(cudaGetDeviceCount(&dev_count));
  for (int device_id : local_gpu_device_id_list) {
    if (device_id >= dev_count) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Invalid device id: " + std::to_string(device_id));
    }
  }

#ifndef ENABLE_INFERENCE
  HCTR_LIB_THROW(nvmlInit_v2());
  CudaCPUDeviceContext::init_cpu_mapping(device_map.get_device_list());
#endif

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
  if (process_id_ == 0) HCTR_LIB_THROW(ncclGetUniqueId(&nid));
  HCTR_MPI_THROW(MPI_Bcast((void*)&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD));

  HCTR_LIB_THROW(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    context.set_device(local_gpu_device_id_list[i]);
    HCTR_LIB_THROW(
        ncclCommInitRank(&comms[i], global_gpu_count, nid, device_map_.get_global_id(i)));
  }
  HCTR_LIB_THROW(ncclGroupEnd());
#else
  HCTR_LIB_THROW(ncclCommInitAll(comms.data(), local_gpu_device_id_list.size(),
                                 local_gpu_device_id_list.data()));
#endif

  gpu_resources_.resize(local_gpu_count);
#pragma omp parallel num_threads(local_gpu_count)
  {
    size_t id = omp_get_thread_num();
    set_local_gpu(std::make_shared<GPUResource>(local_gpu_device_id_list[id], id,
                                                device_map_.get_global_id(id), replica_uniform_seed,
                                                local_replica_variant_seeds[id], comms[id]),
                  id);
  }

  for (size_t i = 0; i < local_gpu_count; i++) {
    p2p_matrix_.push_back(std::vector<bool>(local_gpu_count, false));
  }

  enable_all_peer_accesses();
  if (all_p2p_enabled() == false) {
    HCTR_LOG_S(WARNING, ROOT) << "Peer-to-peer access cannot be fully enabled." << std::endl;
  }

  all2all_warmup();

  initialize_rmm_resources();
  // int dev_id = 0;
  // cudaGetDevice(&dev_id);
  // HCTR_LOG(INFO, WORLD, "ResourceManagerCore ctor getCurrentDeviceId after rmm_init %d\n",
  // dev_id);
}
ResourceManagerCore::~ResourceManagerCore() {
  if (original_device_resource_.empty()) {
    return;
  }
  auto local_gpu_device_id_list = get_local_gpu_device_id_list();
  CudaDeviceContext context;
  for (size_t i = 0; i < local_gpu_device_id_list.size(); i++) {
    context.set_device(local_gpu_device_id_list[i]);
    rmm::mr::set_current_device_resource(original_device_resource_[i]);
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

const std::shared_ptr<rmm::mr::device_memory_resource>&
ResourceManagerCore::get_device_rmm_device_memory_resource(int local_gpu_id) const {
  auto dev_list = get_local_gpu_device_id_list();
  auto it = std::find(dev_list.begin(), dev_list.end(), local_gpu_id);
  auto index = std::distance(dev_list.begin(), it);
  return memory_resource_[index];
}

}  // namespace HugeCTR
