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

#include "HugeCTR/include/layers/dot_product_layer.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

#include <algorithm>
#include <functional>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32

template <typename T>
__global__ void dot_product_kernel(T** inputs, T* output, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    T tmp = 1;
    for (int i = 0; i < num; i++) {
      tmp *= inputs[i][tid];
    }
    output[tid] = tmp;
  }
}

template <typename T>
__global__ void dot_product_dgrad_kernel(const T* top_grad, T** dgrads, T* fprop_output, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < num; ++i){
      if (0 == fprop_output[tid]){
        dgrads[i][tid] = 0;
      } else {
        T d_input = dgrads[i][tid];
        dgrads[i][tid] = top_grad[tid] * ((float)fprop_output[tid] / d_input);
      }
    }
  }
}

}  // end of namespace

DotProductLayer::DotProductLayer(const std::vector<std::shared_ptr<Tensor<float>>>& in_tensors,
                                 const std::shared_ptr<Tensor<float>>& out_tensor, int device_id)
    : Layer(device_id) {
  try {
    CudaDeviceContext context(get_device_id());

    size_ = in_tensors[0]->get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0]->get_dims();
    if (num_ < 2) {
      CK_THROW_(Error_t::WrongInput, "DotProductLayer needs at least 2 input tensors");
    }
    for (int i = 1; i < num_; i++) {
      if (in_tensors[i]->get_dims().size() != dims.size()) {
        CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.size(); j++) {
        if (in_tensors[i]->get_dims()[j] != dims[j]) {
          CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (int i = 0; i < num_; i++) {
      in_tensors_.emplace_back(in_tensors[i]);
    }
    out_tensors_.emplace_back(out_tensor);

    CK_CUDA_THROW_(cudaMallocHost((void**)(&h_inputs_), num_ * sizeof(float*)));
    CK_CUDA_THROW_(cudaMalloc((void**)(&d_inputs_), num_ * sizeof(float*)));

    CK_CUDA_THROW_(cudaMalloc((void**)(&fprop_output_), out_tensor->get_size()));

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

DotProductLayer::~DotProductLayer() {
  try {
    if (h_inputs_) {
      CK_CUDA_THROW_(cudaFreeHost(h_inputs_));
    }
    if (d_inputs_) {
      CK_CUDA_THROW_(cudaFree(d_inputs_));
    }
    if (fprop_output_){
      CK_CUDA_THROW_(cudaFree(fprop_output_));
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

void DotProductLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (int i = 0; i < num_; i++) {
      h_inputs_[i] = in_tensors_[i]->get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_, (void*)h_inputs_, num_ * sizeof(float*),
                                   cudaMemcpyHostToDevice, stream));
    initialized_ = true;
  }
  float* output = out_tensors_[0]->get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  dot_product_kernel<<<gridSize, blockSize, 0, stream>>>(d_inputs_, output, size_, num_);

  CK_CUDA_THROW_(cudaMemcpyAsync((void*)fprop_output_, (void*)output, out_tensors_[0]->get_size(),
                                 cudaMemcpyDeviceToDevice, stream));
}

void DotProductLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (int i = 0; i < num_; i++) {
      h_inputs_[i] = in_tensors_[i]->get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_, (void*)h_inputs_, num_ * sizeof(float*),
                                   cudaMemcpyHostToDevice, stream));
    initialized_ = true;
  }
  float* output = out_tensors_[0]->get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  dot_product_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(output, d_inputs_, fprop_output_, size_, num_);
}

}  // namespace HugeCTR
