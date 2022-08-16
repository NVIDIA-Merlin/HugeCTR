/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>

#include "common/check.h"
#include "lookup/impl/reorder_kernel.h"

namespace sok {

template <typename EmbeddingType>
__global__ void reorderKernel(const size_t EmbeddingDimension, EmbeddingType const *inputs,
                              int32_t const *indices, EmbeddingType *outputs, size_t num_keys) {
  size_t thread_cnt = blockDim.x * gridDim.x;
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t items = num_keys * EmbeddingDimension;
  for (size_t i = thread_idx; i < items; i += thread_cnt) {
    size_t row = indices[i / EmbeddingDimension];
    size_t col = i % EmbeddingDimension;
    outputs[row * EmbeddingDimension + col] = inputs[i];
  }
}

template <typename EmbeddingType>
__global__ void gatherExKernel(const size_t EmbeddingDimension, EmbeddingType const *inputs,
                               int32_t const *indices, EmbeddingType *outputs, size_t num_keys) {
  size_t thread_cnt = blockDim.x * gridDim.x;
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t items = num_keys * EmbeddingDimension;
  for (size_t i = thread_idx; i < items; i += thread_cnt) {
    size_t row = indices[i / EmbeddingDimension];
    size_t col = i % EmbeddingDimension;
    outputs[i] = inputs[row * EmbeddingDimension + col];
  }
}

template <typename DType>
void ReorderLauncher<DType>::initialize() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename DType>
void ReorderLauncher<DType>::operator()(const void *embedding, size_t num_keys, size_t dimension,
                                        const void *order, void *output, cudaStream_t stream) {
  const DType *t_embedding = reinterpret_cast<const DType *>(embedding);
  const int32_t *t_order = reinterpret_cast<const int32_t *>(order);
  DType *t_output = reinterpret_cast<DType *>(output);

  dim3 grid_dim(2 * sm_count_);
  dim3 block_dim(1024ul);
  reorderKernel<DType>
      <<<grid_dim, block_dim, 0, stream>>>(dimension, t_embedding, t_order, t_output, num_keys);

  CUDACHECK(cudaGetLastError());
}

template <typename DType>
void GatherExLauncher<DType>::initialize() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename DType>
void GatherExLauncher<DType>::operator()(const void *grads, size_t num_keys, size_t dimension,
                                         const void *indices, void *output, cudaStream_t stream) {
  const DType *t_grads = reinterpret_cast<const DType *>(grads);
  const int32_t *t_indices = reinterpret_cast<const int32_t *>(indices);
  DType *t_output = reinterpret_cast<DType *>(output);

  dim3 grid_dim(2 * sm_count_);
  dim3 block_dim(1024ul);
  gatherExKernel<DType>
      <<<grid_dim, block_dim, 0, stream>>>(dimension, t_grads, t_indices, t_output, num_keys);

  CUDACHECK(cudaGetLastError());
}

template class ReorderLauncher<float>;
template class ReorderLauncher<__half>;
template class GatherExLauncher<float>;
template class GatherExLauncher<__half>;

}  // namespace sok
