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

#include "HugeCTR/include/layers/multiply_layer.hpp"

#include "HugeCTR/include/layers/element_wise_function.hpp"

#include <algorithm>
#include <functional>
#include "HugeCTR/include/utils.hpp"
#ifndef NDEBUG
#include <iostream>
#endif
 
namespace HugeCTR {
 
namespace {

template<typename T>
__global__ void multiply_kernel(const T * input, 
                                const T * weight, 
                                T * output, 
                                const int batch_size, 
                                const int vector_length) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = (size_t)batch_size * vector_length;

  if(tid < size) {
    output[tid] = input[tid] * weight[tid % vector_length];
  }
}

template<typename T>
__global__ void multiply_wgrad_kernel(const T * top_grad,
                                      const T * input,
                                      T * wgrad,
                                      const int batch_size,
                                      const int vector_length) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = (size_t)batch_size * vector_length;

  if(tid < size) {
    wgrad[tid] = top_grad[tid] * input[tid];
  }
}

template<typename T>
__global__ void multiply_dgrad_kernel(const T * top_grad,
                                      const T * weight,
                                      T * dgrad,
                                      const int batch_size,
                                      const int vector_length) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = (size_t)batch_size * vector_length;

  if(tid < size) {
    dgrad[tid] = top_grad[tid] * weight[tid % vector_length];
  }
}

}

MultiplyLayer::MultiplyLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                            const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                            const std::shared_ptr<Tensor<float>>& in_tensor,
                            const std::shared_ptr<Tensor<float>>& out_tensor, 
                            int device_id)
     : Layer(device_id) {
  try {
    CudaDeviceContext context(get_device_id());

    auto in_dims = in_tensor->get_dims();
    if(in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be multiplied");
    }
    if(in_tensor->get_format() != TensorFormat_t::HW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed for multiply layer");
    }
    auto out_dims = out_tensor->get_dims();
    if(out_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "only 2D tensors can be set as the result of multiply layer");
    }
    if(out_tensor->get_format() != TensorFormat_t::HW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed for multiply layer");
    }
    if(get_size_from_dims(in_dims) != get_size_from_dims(out_dims)) {
      std::cout << "in_size=" << get_size_from_dims(in_dims) << ", out_size=" << get_size_from_dims(out_dims) << std::endl;
      CK_THROW_(Error_t::WrongInput, "the size of the output is not the same as the input");   
    }

    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);

    weights_.emplace_back(new Tensor<float>({1, in_dims[1]}, weight_buff, TensorFormat_t::HW));
    wgrad_.emplace_back(new Tensor<float>(out_dims, wgrad_buff, TensorFormat_t::HW));

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}
 
void MultiplyLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  float* input = in_tensors_[0]->get_ptr();
  float * weight = weights_[0]->get_ptr();
  float* output = out_tensors_[0]->get_ptr();
  int batch_size = in_tensors_[0]->get_dims()[0];
  int vector_length = in_tensors_[0]->get_dims()[1];
  size_t size = (size_t)batch_size * vector_length;

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size + blockSize.x - 1)/blockSize.x, 1, 1);
  multiply_kernel<<<gridSize, blockSize, 0, stream>>>(input, weight, output, 
                                                      batch_size, vector_length);
}
 
void MultiplyLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  float* weight = weights_[0]->get_ptr();
  float* wgrad = wgrad_[0]->get_ptr();
  float* input = in_tensors_[0]->get_ptr();
  float* output = out_tensors_[0]->get_ptr();
  int batch_size = in_tensors_[0]->get_dims()[0];
  int vector_length = in_tensors_[0]->get_dims()[1];
  size_t size = (size_t)batch_size * vector_length;

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size + blockSize.x - 1)/blockSize.x, 1, 1);
  multiply_wgrad_kernel<<<gridSize, blockSize, 0, stream>>>(output, input, wgrad, 
                                                            batch_size, vector_length);
  multiply_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(output, weight, input, 
                                                            batch_size, vector_length);
}
 
}  // namespace HugeCTR
 