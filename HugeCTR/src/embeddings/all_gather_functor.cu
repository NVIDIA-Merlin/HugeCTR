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
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/**
 * collection communication: all_gather.
 * @param send_count the count of elements will be sent.
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename Type>
void SparseEmbeddingFunctors::all_gather(size_t send_count, const Tensors2<Type> &send_tensors,
                                         Tensors2<Type> &recv_tensors,
                                         const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

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

  // for multi GPUs, use NCCL to do All-Gather
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_manager.get_local_gpu(id);
      CK_NCCL_THROW_(ncclAllGather(send_tensors[id].get_ptr(),  // send buff
                                   recv_tensors[id].get_ptr(),  // recv buff
                                   send_count, type, local_gpu->get_nccl(),
                                   local_gpu->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  // for single GPU, just do memcpyD2D
  else {  // total_gpu_count == 1
    const auto &local_gpu = resource_manager.get_local_gpu(0);
    CudaDeviceContext context(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0].get_ptr(), send_tensors[0].get_ptr(),
                                   send_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                   local_gpu->get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::all_gather<float>(size_t send_count,
                                                         const Tensors2<float> &send_tensors,
                                                         Tensors2<float> &recv_tensors,
                                                         const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all_gather<__half>(size_t send_count,
                                                          const Tensors2<__half> &send_tensors,
                                                          Tensors2<__half> &recv_tensors,
                                                          const ResourceManager &resource_manager);

}  // namespace HugeCTR