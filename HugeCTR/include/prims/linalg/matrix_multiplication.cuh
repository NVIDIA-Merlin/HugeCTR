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

#pragma once

#include <cuda_utils.cuh>

namespace MLCommon {
namespace LinAlg {

/**
 * out product of two vectors
 * @param out_mat: hxw
 * @param vec_a: hx1
 * @param vec_b: 1xw
 */
 __global__ void mm_1d(float* out_mat, const float* vec_a, int h, const float* vec_b,
  int w) {
const int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid < h * w) {
const int col = tid % w;
const int row = tid / w;
out_mat[tid] = vec_a[row] * vec_b[col];
}
}

};  // end namespace LinAlg
};  // end namespace MLCommon
