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

#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace hybrid_embedding {

namespace {

template <typename emtype>
__global__ void sgd_global_update_kernel(const emtype *__restrict__ gradients,
                                         float *__restrict__ embedding_vectors,
                                         uint32_t embedding_vec_size,
                                         const float *__restrict__ lr_ptr, const float scale) {
  int bid = blockIdx.x;   // block = one vector
  int tid = threadIdx.x;  // thread = one element in a vector

  float lr = __ldg(lr_ptr) / scale;

  /// TODO: vectorization possible?
  embedding_vectors[bid * embedding_vec_size + tid] -=
      lr * TypeConvertFunc<float, emtype>::convert(gradients[bid * embedding_vec_size + tid]);
}

template <typename emtype, typename LambdaNum, typename LambdaIdx>
__global__ void sgd_atomic_update_kernel(const emtype *__restrict__ gradients,
                                         float *__restrict__ embedding_vectors,
                                         LambdaNum get_num_indices,
                                         LambdaIdx get_index,
                                         uint32_t embedding_vec_size,
                                         const float *__restrict__ lr_ptr, const float scale) {
  const uint32_t num_indices = get_num_indices();

  float lr = __ldg(lr_ptr) / scale;

  for (uint32_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
    auto index = get_index(i);

    atomicAdd(embedding_vectors + index * embedding_vec_size + threadIdx.x,
              -lr * TypeConvertFunc<float, emtype>::convert(
                        gradients[i * embedding_vec_size + threadIdx.x]));
  }
}

}  // namespace

template <typename dtype, typename emtype>
void sgd_global_update(const emtype *gradients, float *embedding_vectors,
                       dtype num_embedding_vectors, uint32_t embedding_vec_size, float *lr_ptr,
                       float scale, cudaStream_t stream) {
  if (num_embedding_vectors < 1) return;
  sgd_global_update_kernel<<<num_embedding_vectors, embedding_vec_size, 0, stream>>>(
      gradients, embedding_vectors, embedding_vec_size, lr_ptr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename emtype, typename LambdaNum, typename LambdaIdx>
void sgd_atomic_update(const emtype *gradients, float *embedding_vectors,
                       LambdaNum get_num_indices, LambdaIdx get_index, uint32_t n_blocks,
                       uint32_t embedding_vec_size, float *lr_ptr, float scale,
                       cudaStream_t stream) {
  // Note: currently taking the number of blocks as an argument but we can also compute it here with
  // some heuristics if we think it's better
  sgd_atomic_update_kernel<<<n_blocks, embedding_vec_size, 0, stream>>>(
      gradients, embedding_vectors, get_num_indices, get_index, embedding_vec_size, lr_ptr, scale);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

}  // namespace hybrid_embedding

}  // namespace HugeCTR
