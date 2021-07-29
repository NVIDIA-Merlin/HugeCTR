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

#include <algorithm>
#include <functional>
#include <layers/reduce_mean_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T, size_t length>
__device__ int array_length(T (&arr)[length]) {
  return length;
}

// this kernel can support dims_size=1/2/3
template <typename T, typename... Args>
__global__ void reduce_mean_kernel(const T* input, T* output, int axis, Args... args) {
  size_t in_dims[] = {args...};
  int dims_size = array_length(in_dims);
  float local_sum = 0.0f;
  int fdims = 1;

  if (axis == 0) {  // block_num = dim1 * dim2, do dim0 number of elements reduction in one block
    if (dims_size == 1) {  // dims_size == 1
      for (int tid = threadIdx.x; tid < in_dims[0]; tid += blockDim.x) {
        local_sum += input[tid];
      }
    } else if (dims_size == 2) {  // dims_size == 2
      for (int tid = threadIdx.x; tid < in_dims[0]; tid += blockDim.x) {
        local_sum += input[tid * in_dims[1] + blockIdx.x];
      }
      fdims = in_dims[0];
    } else if (dims_size == 3) {  // dims_size == 3
      for (int tid = threadIdx.x; tid < in_dims[0]; tid += blockDim.x) {
        local_sum += input[tid * (in_dims[1] * in_dims[2]) + blockIdx.x];
      }
      fdims = in_dims[0];
    }
  } else if (axis ==
             1) {  // block_num = dim0 * dim2, do dim1 number of elements reduction in one block
    if (dims_size == 2) {  // dims_size == 2
      for (int tid = threadIdx.x; tid < in_dims[1]; tid += blockDim.x) {
        local_sum += input[blockIdx.x * in_dims[1] + tid];
      }
    } else if (dims_size == 3) {  // dims_size == 3
      for (int tid = threadIdx.x; tid < in_dims[1]; tid += blockDim.x) {
        local_sum += input[blockIdx.x / in_dims[2] * (in_dims[1] * in_dims[2]) + tid * in_dims[2] +
                           blockIdx.x % in_dims[2]];
      }
    }
    fdims = in_dims[1];
  } else if (axis ==
             2) {  // block_num = dim0 * dim1, do dim2 number of elements reduction in one block
    for (int tid = threadIdx.x; tid < in_dims[2]; tid += blockDim.x) {
      local_sum += input[blockIdx.x * in_dims[2] + tid];
    }
    fdims = in_dims[2];
  }

  local_sum = blockReduceSum(local_sum);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = local_sum / fdims;
  }
}

template <typename T, typename... Args>
__global__ void reduce_mean_dgrad_kernel(const T* top_grad, T* dgrad, int axis, Args... args) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t in_dims[] = {args...};
  int dims_size = array_length(in_dims);

  if (axis == 0) {
    if (dims_size == 1) {  // dims_size == 1
      if (tid < in_dims[0]) {
        dgrad[tid] = top_grad[0];
      }
    } else if (dims_size == 2) {  // dims_size == 2
      if (tid < (in_dims[0] * in_dims[1])) {
        dgrad[tid] = top_grad[tid % in_dims[1]] / in_dims[0];
      }
    } else if (dims_size == 3) {  // dims_size == 3
      if (tid < (in_dims[0] * in_dims[1] * in_dims[2])) {
        int dim1_index = tid % (in_dims[1] * in_dims[2]) / in_dims[2];
        int dim2_index = tid % in_dims[2];
        dgrad[tid] = top_grad[dim1_index * in_dims[2] + dim2_index] / in_dims[0];
      }
    }
  } else if (axis == 1) {
    if (dims_size == 2) {  // dims_size == 2
      if (tid < (in_dims[0] * in_dims[1])) {
        dgrad[tid] = top_grad[tid / in_dims[1]] / in_dims[1];
      }
    } else if (dims_size == 3) {  // dims_size == 3
      if (tid < (in_dims[0] * in_dims[1] * in_dims[2])) {
        int dim0_index = tid / (in_dims[1] * in_dims[2]);
        int dim2_index = tid % in_dims[2];
        dgrad[tid] = top_grad[dim0_index * in_dims[2] + dim2_index] / in_dims[1];
      }
    }
  } else if (axis == 2) {
    int dim0_index = tid / (in_dims[1] * in_dims[2]);
    int dim1_index = tid % (in_dims[1] * in_dims[2]) / in_dims[2];
    dgrad[tid] = top_grad[dim0_index * in_dims[1] + dim1_index] / in_dims[2];
  }
}

}  // end of namespace
template <typename T>
ReduceMeanLayer<T>::ReduceMeanLayer(
    const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, int axis,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), axis_(axis) {
  try {
    // error input checking
    const auto& in_dims = in_tensor.get_dimensions();
    for (auto i : in_dims) {
      if (i == 0) {
        CK_THROW_(Error_t::WrongInput, "The input dims can not be 0");
      }
    }
    if (axis >= (int)(in_dims.size()) || axis < 0) {
      CK_THROW_(Error_t::WrongInput, "The axis is overflow");
    }

    std::vector<size_t> out_dims(in_dims.size());
    for (int i = 0; i < (int)(in_dims.size()); i++) {
      if (i == axis) {
        out_dims[i] = 1;
      } else {
        out_dims[i] = in_dims[i];
      }
    }

    blobs_buff->reserve(out_dims, &out_tensor);
    out_tensors_.push_back(out_tensor);
    in_tensors_.push_back(in_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}
template <typename T>
void ReduceMeanLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* input = in_tensors_[0].get_ptr();
  T* output = out_tensors_[0].get_ptr();
  auto in_dims = in_tensors_[0].get_dimensions();
  auto out_dims = out_tensors_[0].get_dimensions();

  int block_num = 1;
  for (auto dim : out_dims) {
    block_num *= dim;
  }

  dim3 blockSize(256, 1, 1);
  dim3 gridSize(block_num, 1, 1);
  if (in_dims.size() == 1) {
    reduce_mean_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(input, output, axis_,
                                                                           in_dims[0]);
  } else if (in_dims.size() == 2) {
    reduce_mean_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(input, output, axis_,
                                                                           in_dims[0], in_dims[1]);
  } else if (in_dims.size() == 3) {
    reduce_mean_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
        input, output, axis_, in_dims[0], in_dims[1], in_dims[2]);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void ReduceMeanLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  T* input = in_tensors_[0].get_ptr();
  T* output = out_tensors_[0].get_ptr();
  auto in_dims = in_tensors_[0].get_dimensions();

  int size = 1;
  for (auto dim : in_dims) {
    size *= dim;
  }

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1, 1);
  if (in_dims.size() == 1) {
    reduce_mean_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(output, input,
                                                                                 axis_, in_dims[0]);
  } else if (in_dims.size() == 2) {
    reduce_mean_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
        output, input, axis_, in_dims[0], in_dims[1]);
  } else if (in_dims.size() == 3) {
    reduce_mean_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
        output, input, axis_, in_dims[0], in_dims[1], in_dims[2]);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class ReduceMeanLayer<float>;

}  // namespace HugeCTR
