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
#include <common.hpp>

namespace HugeCTR {

template <typename T>
struct TypeFunc;

template <>
struct TypeFunc<float> {
  static __forceinline__ __device__ float zero() { return 0.f; }
  static __forceinline__ __device__ float add(float a, float b) { return a + b; }
};

template <>
struct TypeFunc<int> {
  static __forceinline__ __device__ float zero() { return 0; }
  static __forceinline__ __device__ float add(int a, int b) { return a + b; }
};

template <>
struct TypeFunc<__half> {
  static __forceinline__ __device__ __half zero() { return __float2half(0.0f); }
  static __forceinline__ __device__ __half add(__half a, __half b) { return __hadd(a, b); }
};

template <>
struct TypeFunc<__half2> {
  static __forceinline__ __device__ __half2 zero() { return __float2half2_rn(0.0f); }
  static __forceinline__ __device__ __half2 add(__half2 a, __half2 b) { return __hadd2(a, b); }
};

template <typename TOUT, typename TIN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<__half, float> {
  static __forceinline__ __device__ __half convert(float val) { return __float2half(val); }
};

template <>
struct TypeConvertFunc<float, __half> {
  static __forceinline__ __device__ float convert(__half val) { return __half2float(val); }
};

template <>
struct TypeConvertFunc<float, float> {
  static __forceinline__ __device__ float convert(float val) { return val; }
};

template <>
struct TypeConvertFunc<float, long long> {
  static __forceinline__ __device__ float convert(long long val) { return static_cast<float>(val); }
};

template <>
struct TypeConvertFunc<float, unsigned int> {
  static __forceinline__ __device__ float convert(unsigned int val) {
    return static_cast<float>(val);
  }
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  const unsigned int FULL_MASK = 0xffffffff;
  for (int i = warpSize / 2; i > 0; i >>= 1)
    val = TypeFunc<T>::add(val, __shfl_xor_sync(FULL_MASK, val, i));
  return val;
}

/* Calculate the sum of all elements in a block */
/* Note that the max block size to use this function is 1024 */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (blockDim.x > warpSize) {
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < ((blockDim.x - 1) >> 5) + 1) ? shared[lane] : TypeFunc<T>::zero();
    val = warpReduceSum(val);
  }

  return val;
}

template <typename T>
__global__ void initialize_array(T* array, size_t num_elements, T value) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    array[tid] = value;
  }
}

template <typename TIN, typename TOUT, typename Lambda>
__global__ void transform_array(const TIN* in, TOUT* out, size_t num_elements, Lambda op) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = op(in[tid]);
  }
}

template <typename TIN, typename TOUT>
__global__ void convert_array(TOUT* out, const TIN* in, size_t num_elements) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = TypeConvertFunc<TOUT, TIN>::convert(__ldg(in + tid));
  }
}

}  // namespace HugeCTR
