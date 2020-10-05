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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#ifndef NCCL_A2A
#ifndef ENABLE_MPI
#include "HugeCTR/include/faster_gossip_comm/FasterGossipComm.h"
#else
#include "HugeCTR/include/faster_gossip_comm/FasterGossipCommMulti.h"
#endif
#endif

namespace HugeCTR {

#ifndef NCCL_A2A
#ifdef ENABLE_MPI

template <typename Type>
void SparseEmbeddingFunctors::all2all_init_backward(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
    const ResourceManager &resource_manager) {
  using transfer_plan_t =
      typename GossipComm::FasterGossipCommMultiAll2AllTraits<Type>::transfer_plan_t;
  transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
  size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node
  std::vector<int> device_list = resource_manager.get_local_gpu_device_id_list();
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  if (local_gpu_count != plan_gpu_count) {
    std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
              << std::endl;
    CK_THROW_(Error_t::WrongInput, "Error: the local device_list doesn't match all2all plan_file");
  }

  int total_rank = 1;
  int my_rank = 0;
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &total_rank));
  if (total_gpu_count != (total_rank * local_gpu_count)) {
    CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
  }
#ifndef NDEBUG
  std::cout << "total_rank=" << total_rank << ", my_rank=" << my_rank
            << ", total_gpu_count=" << total_gpu_count << ", local_gpu_count=" << local_gpu_count
            << std::endl;
#endif

  std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());
#ifndef NDEBUG
  std::cout << "gpu device list: { ";
  for (auto dev : device_ids) {
    std::cout << dev << " ";
  }
  std::cout << "}" << std::endl;
#endif

  // The all2all communication class
  auto faster_gossip_comm = new GossipComm::FasterGossipCommMulti<Type>(
      plan_file, device_ids, total_rank, my_rank, MPI_COMM_WORLD);

  std::vector<Type *> src(local_gpu_count);
  std::vector<Type *> dst(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src[id] = send_tensors[id].get_ptr();
    dst[id] = recv_tensors[id].get_ptr();
  }

  std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                              std::vector<size_t>(total_gpu_count));
  std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                              std::vector<size_t>(total_gpu_count));

  // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t global_id = resource_manager.get_local_gpu(i)->get_global_gpu_id();
    size_t slot_num_per_gpu =
        slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
    size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

    for (size_t j = 0; j < total_gpu_count; j++) {
      recv_table[i][j] = element_per_recv;
    }
  }

  // Fill in sending partition table, ith Topo GPU send to jth global GPU
  for (size_t j = 0; j < total_gpu_count; j++) {
    size_t global_id = j;
    size_t slot_num_per_gpu =
        slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
    size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

    for (size_t i = 0; i < local_gpu_count; i++) {
      send_table[i][j] = element_per_send;
    }
  }

#ifndef NDEBUG
  std::cout << "my_rank=" << my_rank << ", gossip all2all backward send_table:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << send_table[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "my_rank=" << my_rank << ", gossip all2all backward recv_table:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << recv_table[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  faster_gossip_comm->Initialize(src, dst, send_table, recv_table);
  all2all.reset(faster_gossip_comm);

  return;
}

template void SparseEmbeddingFunctors::all2all_init_backward<float>(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_init_backward<__half>(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#else

template <typename Type>
void SparseEmbeddingFunctors::all2all_init_backward(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, Tensors2<Type> &send_tensors, Tensors2<Type> &recv_tensors,
    const ResourceManager &resource_manager) {
  using transfer_plan_t = typename GossipComm::FasterGossipCommAll2AllTraits<Type>::transfer_plan_t;
  transfer_plan_t *transfer_plan = new transfer_plan_t(parse_plan(plan_file.c_str()));
  size_t plan_gpu_count = transfer_plan->num_gpus();  // total number of GPUs in current node

  std::vector<int> device_list = resource_manager.get_local_gpu_device_id_list();
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  if (local_gpu_count != plan_gpu_count) {
    std::cout << "local_gpu_count=" << local_gpu_count << ", plan_gpu_count=" << plan_gpu_count
              << std::endl;
    CK_THROW_(Error_t::WrongInput, "Error: the local device_list doesn't match all2all plan_file");
  }
  std::vector<gossip::gpu_id_t> device_ids(device_list.begin(), device_list.end());
#ifndef NDEBUG
  std::cout << "gpu device list: { ";
  for (auto dev : device_ids) {
    std::cout << dev << " ";
  }
  std::cout << "}" << std::endl;
#endif

  auto faster_gossip_comm = new GossipComm::FasterGossipComm<Type>(plan_file, device_ids);

  // The all2all communication class

  std::vector<Type *> src(local_gpu_count);
  std::vector<Type *> dst(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src[id] = send_tensors[id].get_ptr();
    dst[id] = recv_tensors[id].get_ptr();
  }

  // Fill in partition table, ith Topo GPU to jth Topo GPU
  std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
    for (size_t j = 0; j < local_gpu_count; j++) {
      table[j][i] = element_per_send;
    }
  }

#ifndef NDEBUG
  std::cout << "gossip all2all backward table:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << table[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  faster_gossip_comm->Initialize(src, dst, table);
  all2all.reset(faster_gossip_comm);

  return;
}

template void SparseEmbeddingFunctors::all2all_init_backward<float>(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_init_backward<__half>(
    std::unique_ptr<GossipComm::FasterComm> &all2all, const std::string &plan_file,
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#endif
#endif

}  // namespace HugeCTR