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
void SparseEmbeddingFunctors::get_forward_results(
    size_t memcpy_size, const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
    Tensor2<TypeEmbeddingComp> &embedding_feature, Tensors2<TypeEmbeddingComp> &temp_tensors,
    const ResourceManager &resource_manager) {
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  const auto &local_gpu = resource_manager.get_local_gpu(0);

  CudaDeviceContext context;
  if (total_gpu_count > 1) {
    // nccl allGather
    all_gather(memcpy_size,
               embedding_feature_tensors,  // send
               temp_tensors,               // recv
               resource_manager);

    // memcpy D2H
    context.set_device(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaMemcpyAsync(embedding_feature.get_ptr(), temp_tensors[0].get_ptr(),
                                   total_gpu_count * memcpy_size * sizeof(TypeEmbeddingComp),
                                   cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
  } else {
    context.set_device(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaMemcpyAsync(
        embedding_feature.get_ptr(), embedding_feature_tensors[0].get_ptr(),
        memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost, local_gpu->get_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
  }

  return;
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::get_forward_results(
    size_t memcpy_size, const Tensors2<TypeEmbeddingComp> &embedding_feature_tensors,
    void *const embedding_feature, Tensors2<TypeEmbeddingComp> &temp_tensors,
    const ResourceManager &resource_manager, const bool on_gpu) {
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  const auto &local_gpu = resource_manager.get_local_gpu(0);

  cudaMemcpyKind direction = (on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);

  if (total_gpu_count > 1) {
    // nccl allGather
    all_gather(memcpy_size,
               embedding_feature_tensors,  // send
               temp_tensors,               // recv
               resource_manager);

    // memcpy D2H
    CudaDeviceContext context;
    context.set_device(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaMemcpyAsync(embedding_feature, temp_tensors[0].get_ptr(),
                                   total_gpu_count * memcpy_size * sizeof(TypeEmbeddingComp),
                                   direction, local_gpu->get_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
  } else {
    CudaDeviceContext context;
    context.set_device(local_gpu->get_device_id());
    HCTR_LIB_THROW(cudaMemcpyAsync(embedding_feature, embedding_feature_tensors[0].get_ptr(),
                                   memcpy_size * sizeof(TypeEmbeddingComp), direction,
                                   local_gpu->get_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::get_forward_results<float>(
    size_t memcpy_size, const Tensors2<float> &embedding_feature_tensors,
    Tensor2<float> &embedding_feature, Tensors2<float> &temp_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::get_forward_results<__half>(
    size_t memcpy_size, const Tensors2<__half> &embedding_feature_tensors,
    Tensor2<__half> &embedding_feature, Tensors2<__half> &temp_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::get_forward_results<float>(
    size_t memcpy_size, const Tensors2<float> &embedding_feature_tensors,
    void *const embedding_feature, Tensors2<float> &temp_tensors,
    const ResourceManager &resource_manager, const bool on_gpu);

template void SparseEmbeddingFunctors::get_forward_results<__half>(
    size_t memcpy_size, const Tensors2<__half> &embedding_feature_tensors,
    void *const embedding_feature, Tensors2<__half> &temp_tensors,
    const ResourceManager &resource_manager, const bool on_gpu);

}  // namespace HugeCTR