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

namespace ShuffleKernels {

// Type-agnostic implementations

template <typename VecType>
__device__ inline void vec2arr_void(void* arr, VecType val);
template <typename VecType>
__device__ inline VecType arr2vec_void(const void* arr);

template <>
__device__ inline void vec2arr_void<int4>(void* arr, int4 val) {
  auto int_arr = reinterpret_cast<int*>(arr);
  int_arr[0] = val.x;
  int_arr[1] = val.y;
  int_arr[2] = val.z;
  int_arr[3] = val.w;
}

template <>
__device__ inline int4 arr2vec_void<int4>(const void* arr) {
  auto int_arr = reinterpret_cast<const int*>(arr);
  return {int_arr[0], int_arr[1], int_arr[2], int_arr[3]};
}

template <>
__device__ inline void vec2arr_void<int2>(void* arr, int2 val) {
  auto int_arr = reinterpret_cast<int*>(arr);
  int_arr[0] = val.x;
  int_arr[1] = val.y;
}

template <>
__device__ inline int2 arr2vec_void<int2>(const void* arr) {
  auto int_arr = reinterpret_cast<const int*>(arr);
  return {int_arr[0], int_arr[1]};
}

// Type-aware specializations

template <typename VecType, typename T>
__device__ inline void vec2arr(T* arr, VecType val);
template <>
__device__ inline void vec2arr<int4, __half>(__half* arr, int4 val) {
  vec2arr_void<int4>(reinterpret_cast<void*>(arr), val);
}
template <>
__device__ inline void vec2arr<int4, float>(float* arr, int4 val) {
  vec2arr_void<int4>(reinterpret_cast<void*>(arr), val);
}
template <>
__device__ inline void vec2arr<int4, int>(int* arr, int4 val) {
  vec2arr_void<int4>(reinterpret_cast<void*>(arr), val);
}
template <>
__device__ inline void vec2arr<int2, __half>(__half* arr, int2 val) {
  vec2arr_void<int2>(reinterpret_cast<void*>(arr), val);
}
template <>
__device__ inline void vec2arr<int2, float>(float* arr, int2 val) {
  vec2arr_void<int2>(reinterpret_cast<void*>(arr), val);
}
template <>
__device__ inline void vec2arr<int2, int>(int* arr, int2 val) {
  vec2arr_void<int2>(reinterpret_cast<void*>(arr), val);
}

template <typename VecType, typename T>
__device__ inline VecType arr2vec(const T* arr);
template <>
__device__ inline int4 arr2vec<int4, __half>(const __half* arr) {
  return arr2vec_void<int4>(reinterpret_cast<const void*>(arr));
}
template <>
__device__ inline int4 arr2vec<int4, float>(const float* arr) {
  return arr2vec_void<int4>(reinterpret_cast<const void*>(arr));
}
template <>
__device__ inline int4 arr2vec<int4, int>(const int* arr) {
  return arr2vec_void<int4>(reinterpret_cast<const void*>(arr));
}
template <>
__device__ inline int2 arr2vec<int2, __half>(const __half* arr) {
  return arr2vec_void<int2>(reinterpret_cast<const void*>(arr));
}
template <>
__device__ inline int2 arr2vec<int2, float>(const float* arr) {
  return arr2vec_void<int2>(reinterpret_cast<const void*>(arr));
}
template <>
__device__ inline int2 arr2vec<int2, int>(const int* arr) {
  return arr2vec_void<int2>(reinterpret_cast<const void*>(arr));
}

}  // namespace ShuffleKernels
