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

#include <core23/macros.hpp>
#include <core23/tensor_view.hpp>

namespace HugeCTR {

namespace core23 {

// TODO: existing CUDA primitives and util kernels must be moved to this header file including
// sinusoidal_kernel
// TODO: add VecT

template <typename Type>
__global__ void fill_kernel(Type* data, int64_t num_elements, const Type val) {
  const int64_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t num_threads = blockDim.x * gridDim.x;
  for (int64_t tid = tid_base; tid < num_elements; tid += num_threads) {
    data[tid] = val;
  }
}

template <typename DstType, typename SrcType, typename Op>
__global__ void transform_kernel(DstType* dst, const SrcType* src, int64_t num_elements, Op op) {
  const int64_t tid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t num_threads = blockDim.x * gridDim.x;
  for (int64_t tid = tid_base; tid < num_elements; tid += num_threads) {
    dst[tid] = op(__ldg(src + tid));
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 1> input_tensor,
                            TensorView<BuiltInType, 1> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t x = x_base; x < output_tensor.size(0); x += blockDim.x * gridDim.x) {
    output_tensor[x] = input_tensor[x];
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 2> input_tensor,
                            TensorView<BuiltInType, 2> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t y_base = blockIdx.y * blockDim.y + threadIdx.y;
  for (int64_t y = y_base; y < output_tensor.size(0); y += blockDim.y * gridDim.y) {
    for (int64_t x = x_base; x < output_tensor.size(1); x += blockDim.x * gridDim.x) {
      output_tensor[y][x] = input_tensor[y][x];
    }
  }
}

template <typename BuiltInType>
__global__ void copy_kernel(TensorView<BuiltInType, 3> input_tensor,
                            TensorView<BuiltInType, 3> output_tensor) {
  const int64_t x_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t y_base = blockIdx.y * blockDim.y + threadIdx.y;
  for (int64_t z = 0; z < output_tensor.size(0); z++) {
    for (int64_t y = y_base; y < output_tensor.size(1); y += blockDim.y * gridDim.y) {
      for (int64_t x = x_base; x < output_tensor.size(2); x += blockDim.x * gridDim.x) {
        output_tensor[z][y][x] = input_tensor[z][y][x];
      }
    }
  }
}

}  // namespace core23

}  // namespace HugeCTR