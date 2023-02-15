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
#pragma once

#include <cuda_utils.cuh>

namespace MLCommon {
namespace LinAlg {

/**
 * @brief Compute outer product of two vectors
 *
 * @tparam InType the data type of the 2 input vectors
 * @tparam OutType the data type of the output matrix
 * @param out_mat the output matric of the vectors outer produt
 * @param vec_a the first input vector (hx1)
 * @param h size of vector [vec_a]
 * @param vec_b the second input vector (1xw)
 * @param w size of vector [vec_b]
 */
template <typename InType, typename OutType>
__global__ void mm_1d(OutType *out_mat, const InType *vec_a, int h, const float *vec_b, int w) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < h * w) {
    const int col = tid % w;
    const int row = tid / w;
    out_mat[tid] = vec_a[row] * vec_b[col];
  }
}

};  // end namespace LinAlg
};  // end namespace MLCommon
