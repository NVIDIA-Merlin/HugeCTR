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
void SparseEmbeddingFunctors::get_backward_results(
    size_t id, size_t memcpy_size, const TensorPtrs<TypeEmbeddingComp> &wgrad_tensors,
    TypeEmbeddingComp *wgrad, const GPUResourceGroup &device_resources) {
  CudaDeviceContext context(device_resources[id].get_device_id());
  CK_CUDA_THROW_(cudaMemcpyAsync(wgrad, wgrad_tensors[id]->get_ptr(),
                                 memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost,
                                 device_resources[id].get_stream()));
  CK_CUDA_THROW_(cudaStreamSynchronize(device_resources[id].get_stream()));

  return;
}

template void SparseEmbeddingFunctors::get_backward_results<float>(
    size_t id, size_t memcpy_size, const TensorPtrs<float> &wgrad_tensors, float *wgrad,
    const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::get_backward_results<__half>(
    size_t id, size_t memcpy_size, const TensorPtrs<__half> &wgrad_tensors, __half *wgrad,
    const GPUResourceGroup &device_resources);

}  // namespace HugeCTR