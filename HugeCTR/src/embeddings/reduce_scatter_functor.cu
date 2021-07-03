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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::reduce_scatter(size_t recv_count,
                                             const Tensors2<TypeEmbeddingComp> &send_tensors,
                                             Tensors2<TypeEmbeddingComp> &recv_tensors,
                                             const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  // need to know the type of TypeHashKey here
  ncclDataType_t type;
  switch (sizeof(TypeEmbeddingComp)) {
    case 2:
      type = ncclHalf;
      break;
    case 4:
      type = ncclFloat;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: TypeHashKey not support by now");
  }

  // for multi GPUs, use NCCL to do Reduce-Scatter(supporting multi-node GPU servers)
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_manager.get_local_gpu(id);
      CK_NCCL_THROW_(ncclReduceScatter(send_tensors[id].get_ptr(),  // send buf
                                       recv_tensors[id].get_ptr(),  // recv buff
                                       recv_count, type, ncclSum, local_gpu->get_nccl(),
                                       local_gpu->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  // for single GPU, just do memcpyD2D
  else {  // total_gpu_count == 1
    const auto &local_gpu = resource_manager.get_local_gpu(0);
    CudaDeviceContext context(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0].get_ptr(), send_tensors[0].get_ptr(),
                                   recv_count * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToDevice,
                                   local_gpu->get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::reduce_scatter<float>(
    size_t recv_count, const Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::reduce_scatter<__half>(
    size_t recv_count, const Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR