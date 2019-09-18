/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

template <typename T>
struct TypeFunc;

template <>
struct TypeFunc<float> {
  static __forceinline__ __device__ __host__ float zero() { return 0.f; }
};

template <>
struct TypeFunc<int> {
  static __forceinline__ __device__ __host__ float zero() { return 0; }
};

template <typename T>
__forceinline__ __device__ void atomic_global_sum_div(T x, T* acc, float div) {
  // warp reduce
  const unsigned int FULL_MASK = 0xffffffff;
  const int WARP_DIM = 32;
  for (int i = WARP_DIM / 2; i > 0; i /= 2) {
    x += __shfl_down_sync(FULL_MASK, x, i);
  }
  if (threadIdx.x % WARP_DIM == 0) {
    atomicAdd(acc, (T)(x / div));
  }
  return;
}

template <typename T>
__forceinline__ __device__ void atomic_global_sum(T x, T* acc) {
  // warp reduce
  const unsigned int FULL_MASK = 0xffffffff;
  const int WARP_DIM = 32;
  for (int i = WARP_DIM / 2; i > 0; i /= 2) {
    x += __shfl_down_sync(FULL_MASK, x, i);
  }
  if (threadIdx.x % WARP_DIM == 0) {
    atomicAdd(acc, x);
  }
  return;
}

}  // namespace HugeCTR
