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

namespace HugeCTR {

#ifdef NCCL_A2A
#ifdef ENABLE_MPI

/**
 * nccl all2all communication for backward
 * @param batch_size_per_gpu batch size per GPU
 * @param slot_num slot number
 * @param embedding_vec_size embedding vector size
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 */
template <typename Type>
void SparseEmbeddingFunctors::all2all_backward(size_t batch_size_per_gpu, size_t slot_num,
                                               size_t embedding_vec_size,
                                               const Tensors2<Type> &send_tensors,
                                               Tensors2<Type> &recv_tensors,
                                               const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  size_t num_proc = resource_manager.get_num_process();

  if (total_gpu_count != (num_proc * local_gpu_count)) {
    CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
  }

  std::vector<const Type *> src(local_gpu_count);
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
    size_t global_id = resource_manager.get_local_gpu(i)->get_global_id();
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

  std::vector<std::vector<const Type *>> src_pos(local_gpu_count,
                                                 std::vector<const Type *>(total_gpu_count));
  std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
  // Calculate the src offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t src_offset = 0;
    for (size_t j = 0; j < total_gpu_count; j++) {
      src_pos[i][j] = src[i] + src_offset;
      src_offset += send_table[i][j];
    }
  }
  // Calculate the dst offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t dst_offset = 0;
    for (size_t j = 0; j < total_gpu_count; j++) {
      dst_pos[i][j] = dst[i] + dst_offset;
      dst_offset += recv_table[i][j];
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all backward src_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << src_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "nccl all2all backward dst_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << dst_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  // need to know the Type
  ncclDataType_t type;
  switch (sizeof(Type)) {
    case 2:
      type = ncclHalf;
      break;
    case 4:
      type = ncclFloat;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
  }

  // Do the all2all transfer
  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    const auto &local_gpu = resource_manager.get_local_gpu(i);
    for (size_t j = 0; j < total_gpu_count; j++) {
      CK_NCCL_THROW_(ncclSend(src_pos[i][j], send_table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
      CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], recv_table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
    }
  }
  CK_NCCL_THROW_(ncclGroupEnd());

  return;
}

template void SparseEmbeddingFunctors::all2all_backward<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_backward<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#else

/**
 * nccl all2all communication for backward
 * CAUSION: Only support intra-node all2all currently
 * @param batch_size_per_gpu batch size per GPU
 * @param slot_num_per_gpu slot number for each local GPU
 * @param embedding_vec_size embedding vector size
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 */
template <typename Type>
void SparseEmbeddingFunctors::all2all_backward(size_t batch_size_per_gpu,
                                               const std::vector<size_t> &slot_num_per_gpu,
                                               size_t embedding_vec_size,
                                               const Tensors2<Type> &send_tensors,
                                               Tensors2<Type> &recv_tensors,
                                               const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  // Fill in partition table, ith Topo GPU to jth Topo GPU
  std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
    for (size_t j = 0; j < local_gpu_count; j++) {
      table[j][i] = element_per_send;
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all backward table:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << table[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  std::vector<const Type *> src(local_gpu_count);
  std::vector<Type *> dst(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src[id] = send_tensors[id].get_ptr();
    dst[id] = recv_tensors[id].get_ptr();
  }
  std::vector<std::vector<const Type *>> src_pos(local_gpu_count,
                                                 std::vector<const Type *>(local_gpu_count));
  std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));
  // Calculate the src offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t src_offset = 0;
    for (size_t j = 0; j < local_gpu_count; j++) {
      src_pos[i][j] = src[i] + src_offset;
      src_offset += table[i][j];
    }
  }
  // Calculate the dst offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t dst_offset = 0;
    for (size_t j = 0; j < local_gpu_count; j++) {
      dst_pos[i][j] = dst[i] + dst_offset;
      dst_offset += table[j][i];
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all backward src_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << src_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "nccl all2all backward dst_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << dst_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  // need to know the Type
  ncclDataType_t type;
  switch (sizeof(Type)) {
    case 2:
      type = ncclHalf;
      break;
    case 4:
      type = ncclFloat;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
  }

  // Do the all2all transfer
  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    const auto &local_gpu = resource_manager.get_local_gpu(i);
    for (size_t j = 0; j < local_gpu_count; j++) {
      CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
      CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
    }
  }
  CK_NCCL_THROW_(ncclGroupEnd());

  return;
}

template void SparseEmbeddingFunctors::all2all_backward<float>(
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, const Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_backward<__half>(
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, const Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#endif

#endif

}  // namespace HugeCTR
