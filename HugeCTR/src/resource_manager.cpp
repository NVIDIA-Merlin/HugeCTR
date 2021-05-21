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

#include <collectives/all_reduce_comm.hpp>
#include <random>
#include <resource_manager.hpp>
#include <utils.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
//#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#pragma GCC diagnostic pop

namespace HugeCTR {

std::unordered_map<int, int> CudaCPUDeviceContext::device_id_to_numa_node_;

std::shared_ptr<ResourceManager> ResourceManager::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed,
    DeviceMap::Layout layout) {
  int size = 1, rank = 0;

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &size));
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
#endif

  DeviceMap device_map(visible_devices, rank, layout);

  std::random_device rd;
  if (seed == 0) {
    seed = rd();
  }

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
#endif

  MESSAGE_("Global seed is " + std::to_string(seed));

  CK_NVML_THROW_(nvmlInit_v2());
  CudaCPUDeviceContext::init_cpu_mapping(device_map.get_device_list());

  return std::shared_ptr<ResourceManager>(
      new ResourceManager(size, rank, std::move(device_map), seed));
}

ResourceManager::ResourceManager(int num_process, int process_id, DeviceMap&& device_map,
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
  for (size_t i = 0; i < local_gpu_count; i++) {
    gpu_resources_.emplace_back(new GPUResource(local_gpu_device_id_list[i], i,
                                                device_map_.get_global_id(i), replica_uniform_seed,
                                                local_replica_variant_seeds[i], comms[i]));
  }

  for (size_t i = 0; i < local_gpu_count; i++) {
    p2p_matrix_.push_back(std::vector<bool>(local_gpu_count, false));
  }

  enable_all_peer_accesses();
  if (all_p2p_enabled() == false) {
    MESSAGE_("Peer-to-peer access cannot be fully enabled.");
  }

  all2all_warmup();

  const size_t pool_alloc_size = 256 * 1024 * 1024;

  using dmmr = rmm::mr::device_memory_resource;
  for (size_t i = 0; i < local_gpu_device_id_list.size(); i++) {
    context.set_device(local_gpu_device_id_list[i]);
    base_cuda_mr_.emplace_back(
        std::shared_ptr<rmm::mr::cuda_memory_resource>(new rmm::mr::cuda_memory_resource()));
    memory_resource_.emplace_back(std::shared_ptr<rmm::mr::pool_memory_resource<dmmr>>(
        new rmm::mr::pool_memory_resource<dmmr>(base_cuda_mr_.back().get(), pool_alloc_size)));
    rmm::mr::set_current_device_resource(memory_resource_.back().get());
  }

  if (num_process_ > 1) {
#ifdef ENABLE_MPI
    ib_comm_ = std::make_unique<IbComm>();
    ib_comm_->init(num_process_, local_gpu_count, process_id_, local_gpu_device_id_list);
#endif
  }
}

void ResourceManager::set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision)
{
#ifdef ENABLE_MPI
  ar_comm_ = AllReduceInPlaceComm::create(num_process_, algo, use_mixed_precision, gpu_resources_,
                                          ib_comm_.get());
#else
  ar_comm_ = AllReduceInPlaceComm::create(num_process_, algo, use_mixed_precision, gpu_resources_);
#endif
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

void ResourceManager::all2all_warmup() {
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

}  // namespace HugeCTR
