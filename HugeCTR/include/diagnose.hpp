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
#include <limits>
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace diagnose {

namespace {

__device__ float atomicMin(float* address, float val) {
  float old = val;
  do {
    val = old;
    old = atomicExch(address, val);
  } while (old < val);
  return old;
}

__device__ float atomicMax(float* address, float val) {
  float old = val;
  do {
    val = old;
    old = atomicExch(address, val);
  } while (old > val);
  return old;
}

template <typename T>
__global__ void count_kernel(const T* arr, int len, float* range) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    float val = TypeConvertFunc<float, T>::convert(arr[i]);
    if (val < 0) {
      atomicMin(range + 0, val);
      atomicMax(range + 1, val);
    } else if (val > 0) {
      atomicMin(range + 2, val);
      atomicMax(range + 3, val);
    }
  }
}

template <typename T>
__global__ void check_kernel(const T* arr, int len, int* flag);

template <>
__global__ void check_kernel<float>(const float* arr, int len, int* flag) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (isnan(arr[i])) atomicAdd(flag, 1);
  }
}

template <>
__global__ void check_kernel(const __half* arr, int len, int* flag) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (__hisnan(arr[i])) {
      atomicAdd(flag, 1);
    }
  }
}

template <typename T>
__global__ void sample_kernel(const T* arr, int len, float* arr_sample, int stride,
                              int max_sample_len) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    if (i % stride == 0) {
      int j = i / stride;
      if (j < max_sample_len) {
        arr_sample[j] = TypeConvertFunc<float, T>::convert(arr[i]);
      }
    }
  }
}

}  // namespace

template <typename T>
void check_and_count_data(const char* category, const T* arr, size_t len,
                          const cudaStream_t& stream) {
  float h_array[4]{0.0f, -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity(), 0.0f};
  int h_flag;
  float* d_array;
  int* d_flag;
  CK_CUDA_THROW_(cudaMalloc(&d_array, sizeof(h_array)));
  CK_CUDA_THROW_(cudaMalloc(&d_flag, sizeof(int)));
  CK_CUDA_THROW_(
      cudaMemcpyAsync(d_array, h_array, sizeof(h_array), cudaMemcpyHostToDevice, stream));
  CK_CUDA_THROW_(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));
  count_kernel<<<160, 1024, 0, stream>>>(arr, len, d_array);
  check_kernel<<<160, 1024, 0, stream>>>(arr, len, d_flag);
  CK_CUDA_THROW_(
      cudaMemcpyAsync(h_array, d_array, sizeof(h_array), cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  std::stringstream ss;
  ss << "Data range for " << category << " [" << h_array[0] << ", " << h_array[1] << "]"
     << ", [" << h_array[2] << ", " << h_array[3] << "]" << std::endl;
  MESSAGE_(ss.str());

  if (h_flag != 0) {
    CK_THROW_(Error_t::DataCheckError, std::string("Nan assert for ") + category + " failed(" +
                                           std::to_string(h_flag) + ").");
  }
  CK_CUDA_THROW_(cudaFree(d_array));
  CK_CUDA_THROW_(cudaFree(d_flag));
}

template <typename T>
void sample_data(const char* category, const T* arr, size_t len, const cudaStream_t& stream) {
  const size_t max_sample_len = 31;
  float h_array[max_sample_len]{0};

  float* d_array;
  CK_CUDA_THROW_(cudaMalloc(&d_array, max_sample_len * sizeof(float)));
  CK_CUDA_THROW_(cudaMemsetAsync(d_array, 0, max_sample_len * sizeof(float), stream));
  sample_kernel<<<160, 1024, 0, stream>>>(arr, len, d_array, len / max_sample_len, max_sample_len);
  CK_CUDA_THROW_(cudaMemcpyAsync(h_array, d_array, max_sample_len * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  std::stringstream ss;
  ss << "Data sampling for " << category << " [";
  for (size_t i = 0; i < min(max_sample_len, len); i++) {
    if (i != 0) ss << ",";
    ss << h_array[i];
  }
  ss << "]" << std::endl;
  MESSAGE_(ss.str());

  CK_CUDA_THROW_(cudaFree(d_array));
}

}  // namespace diagnose

}  // namespace HugeCTR