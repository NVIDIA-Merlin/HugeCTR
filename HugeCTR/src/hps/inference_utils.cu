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

#include <hps/inference_utils.hpp>

namespace HugeCTR {

// Kernels to combine the value buffer
__global__ void merge_emb_vec(float* d_output_emb_vec, const float* d_missing_emb_vec,
                              const uint64_t* d_missing_index, const size_t len,
                              const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

// Kernels to fill the default value to the output buffer
__global__ void fill_default_emb_vec(float* d_output_emb_vec, const float default_emb_vec,
                                     const uint64_t* d_missing_index, const size_t len,
                                     const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = default_emb_vec;
  }
}

// Kernels to decompress the value buffer
__global__ void decompress_emb_vec(const float* d_src_emb_vec, const uint64_t* d_src_index,
                                   float* d_dst_emb_vec, const size_t len,
                                   const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t dst_emb_vec = idx / emb_vec_size;
    size_t dst_float = idx % emb_vec_size;
    size_t src_emb_vec = d_src_index[dst_emb_vec];
    d_dst_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_src_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
                         const uint64_t* d_missing_index_ptr, const size_t missing_len,
                         const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, d_vals_retrieved_ptr, d_missing_index_ptr, missing_len, emb_vec_size);
}

void fill_default_emb_vec_async(float* d_vals_merge_dst_ptr, const float default_emb_vec,
                                const uint64_t* d_missing_index_ptr, const size_t missing_len,
                                const size_t emb_vec_size, const size_t BLOCK_SIZE,
                                cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  fill_default_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, default_emb_vec, d_missing_index_ptr, missing_len, emb_vec_size);
}

void decompress_emb_vec_async(const float* d_unique_src_ptr, const uint64_t* d_unique_index_ptr,
                              float* d_decompress_dst_ptr, const size_t decompress_len,
                              const size_t emb_vec_size, const size_t BLOCK_SIZE,
                              cudaStream_t stream) {
  if (decompress_len == 0) {
    return;
  }
  size_t decompress_len_in_float = decompress_len * emb_vec_size;
  decompress_emb_vec<<<((decompress_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_unique_src_ptr, d_unique_index_ptr, d_decompress_dst_ptr, decompress_len, emb_vec_size);
}

}  // namespace HugeCTR
