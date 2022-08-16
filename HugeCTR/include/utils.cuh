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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <common.hpp>

namespace HugeCTR {

template <typename T>
struct TypeFunc;

template <>
struct TypeFunc<float> {
  static __forceinline__ __device__ float zero() { return 0.f; }
  static __forceinline__ __device__ float add(float a, float b) { return a + b; }
  static __forceinline__ __device__ float min() { return -1e20f; }
};

template <>
struct TypeFunc<int> {
  static __forceinline__ __device__ float zero() { return 0; }
  static __forceinline__ __device__ float add(int a, int b) { return a + b; }
};

template <>
struct TypeFunc<unsigned int> {
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

template <>
struct TypeConvertFunc<int, long long> {
  static __forceinline__ __device__ int convert(long long val) { return static_cast<int>(val); }
};

template <>
struct TypeConvertFunc<int, unsigned int> {
  static __forceinline__ __device__ int convert(unsigned int val) { return static_cast<int>(val); }
};
template <typename IntType>
constexpr __host__ __device__ __inline__ IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ __inline__ IntType alignTo(IntType a, IntType b) {
  return ceildiv(a, b) * b;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  const unsigned int FULL_MASK = 0xffffffff;
  for (int i = warpSize / 2; i > 0; i >>= 1)
    val = TypeFunc<T>::add(val, __shfl_xor_sync(FULL_MASK, val, i));
  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  const unsigned int FULL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = warpSize / 2; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FULL_MASK, val, mask, 32));
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
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp
  if (blockDim.x > warpSize) {
    if (lane == 0)  // record in-warp maxx by warp Idx
      shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < ((blockDim.x - 1 >> 5) + 1)) ? shared[lane] : TypeFunc<T>::min();
    val = warpReduceMax<T>(val);
  }

  return val;
}

template <typename T>
__global__ void initialize_array(T *array, size_t num_elements, T value) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    array[tid] = value;
  }
}

template <typename TIN, typename TOUT, typename Lambda>
__global__ void transform_array(const TIN *in, TOUT *out, size_t num_elements, Lambda op) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = op(in[tid]);
  }
}

template <typename TIN, typename TOUT>
__global__ void convert_array(TOUT *out, const TIN *in, size_t num_elements) {
  const size_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t num_threads = blockDim.x * gridDim.x;
  for (size_t tid = tid_base; tid < num_elements; tid += num_threads) {
    out[tid] = TypeConvertFunc<TOUT, TIN>::convert(__ldg(in + tid));
  }
}

namespace unique_key_kernels {
// for onehot
template <typename TypeKey>
__global__ void data_to_unique_categories_kernel(TypeKey *__restrict__ data,
                                                 const TypeKey *__restrict__ embedding_offsets,
                                                 int num_tables, int num_data) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_data;
       idx += blockDim.x * gridDim.x) {
    data[idx] = data[idx] + embedding_offsets[idx % num_tables];
  }
}

template <typename TypeKey>
__global__ void data_to_unique_categories_align2_kernel(
    TypeKey *__restrict__ data, const TypeKey *__restrict__ embedding_offsets, int num_tables,
    int num_data) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_data;
       idx += blockDim.x * gridDim.x) {
    uint2 load_data = reinterpret_cast<uint2 *>(data)[idx];
    uint2 load_embedding_offsets =
        reinterpret_cast<const uint2 *>(embedding_offsets)[idx % num_tables];

    load_data.x += load_embedding_offsets.x;
    load_data.y += load_embedding_offsets.y;
    reinterpret_cast<uint2 *>(data)[idx] = load_data;
  }
}

// for multihot
template <typename TypeKey>
__global__ void data_to_unique_categories_kernel(TypeKey *__restrict__ values,
                                                 const TypeKey *__restrict__ rowoffsets,
                                                 const TypeKey *__restrict__ embedding_offsets,
                                                 int num_tables, int num_rowoffsets) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_rowoffsets) {
    TypeKey offset = embedding_offsets[tid % num_tables];
    for (int i = rowoffsets[tid]; i < rowoffsets[tid + 1]; ++i) {
      values[i] += offset;
    }
  }
}
}  // namespace unique_key_kernels

}  // namespace HugeCTR
