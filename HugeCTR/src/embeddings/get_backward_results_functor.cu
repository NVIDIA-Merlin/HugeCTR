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

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::get_backward_results(size_t id, size_t memcpy_size,
                                                   const Tensors2<TypeEmbeddingComp> &wgrad_tensors,
                                                   Tensor2<TypeEmbeddingComp> &wgrad,
                                                   const ResourceManager &resource_manager) {
  const auto &local_gpu = resource_manager.get_local_gpu(id);
  CudaDeviceContext context(local_gpu->get_device_id());
  HCTR_LIB_THROW(cudaMemcpyAsync(wgrad.get_ptr(), wgrad_tensors[id].get_ptr(),
                                 memcpy_size * sizeof(TypeEmbeddingComp), cudaMemcpyDeviceToHost,
                                 local_gpu->get_stream()));
  HCTR_LIB_THROW(cudaStreamSynchronize(local_gpu->get_stream()));

  return;
}

template void SparseEmbeddingFunctors::get_backward_results<float>(
    size_t id, size_t memcpy_size, const Tensors2<float> &wgrad_tensors, Tensor2<float> &wgrad,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::get_backward_results<__half>(
    size_t id, size_t memcpy_size, const Tensors2<__half> &wgrad_tensors, Tensor2<__half> &wgrad,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR