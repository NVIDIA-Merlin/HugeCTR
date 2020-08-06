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

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::get_forward_results(
    size_t memcpy_size, const TensorPtrs<TypeEmbeddingComp> &embedding_feature_tensors,
    TypeEmbeddingComp *embedding_feature, TensorPtrs<TypeEmbeddingComp> &temp_tensors,
    const GPUResourceGroup &device_resources) {
  CudaDeviceContext context;
  size_t local_gpu_count = device_resources.size();
  size_t total_gpu_count = device_resources.get_total_gpu_count();

  if (total_gpu_count > 1) {
    // nccl allGather
    all_gather(memcpy_size,
               embedding_feature_tensors,  // send
               temp_tensors,               // recv
               device_resources);

    // memcpy D2H
    context.set_device(device_resources[0].get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, temp_tensors[0]->get_ptr(),
                                   total_gpu_count * memcpy_size * sizeof(TypeEmbeddingComp),
                                   cudaMemcpyDeviceToHost, device_resources[0].get_stream()));
    CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[0].get_stream()));
  } else {
    context.set_device(device_resources[0].get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(embedding_feature, embedding_feature_tensors[0]->get_ptr(),
                                   memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost,
                                   device_resources[0].get_stream()));
    CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[0].get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::get_forward_results<float>(
    size_t memcpy_size, const TensorPtrs<float> &embedding_feature_tensors,
    float *embedding_feature, TensorPtrs<float> &temp_tensors,
    const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::get_forward_results<__half>(
    size_t memcpy_size, const TensorPtrs<__half> &embedding_feature_tensors,
    __half *embedding_feature, TensorPtrs<__half> &temp_tensors,
    const GPUResourceGroup &device_resources);

}  // namespace HugeCTR