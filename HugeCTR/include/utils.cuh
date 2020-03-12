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

inline int calc_grid(int t, int b){
  return (t - 1)/b + 1;
}


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

__inline__ __device__ float warpReduceSum(float val) {
  const unsigned int FINAL_MASK = 0xffffffff;
  for (int mask = 16; mask > 0; mask >>= 1) val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  val = warpReduceSum(val);

  return val;
}

template <typename T>
__global__ void initialize_array(T* array, int num_elements, T value) {
  const int tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  for(int tid = tid_base; tid < num_elements; tid += num_threads) {
    array[tid] = value;
  }
}

template <typename T, typename Lambda>
__global__ void transform_array(const T* in, T* out, int num_elements, Lambda op) {
  const int tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  for(int tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = op(in[tid]);
  }
}

template<typename T>
__device__ __forceinline__ T clip(T val, T min,T max){
  val = val < min ? min : val;
  val = val > max ? max : val;
  return val;
}

template<typename T>
__device__ __forceinline__ bool isnan(T val){
  if(val != val){
    return true;
  }
  return false;
}


}  // namespace HugeCTR
