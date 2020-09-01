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
__global__ void dot_product_dgrad_kernel(const T* top_grad, T** dgrads, T* fprop_output, int size,
                                         int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < num; ++i) {
      if (0 == fprop_output[tid]) {
        dgrads[i][tid] = 0;
      } else {
        T d_input = dgrads[i][tid];
        dgrads[i][tid] = top_grad[tid] * ((float)fprop_output[tid] / d_input);
      }
    }
  }
}

}  // end of namespace

DotProductLayer::DotProductLayer(const Tensors2<float>& in_tensors,
                                 const Tensor2<float>& out_tensor,
                                 const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                                 const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      CK_THROW_(Error_t::WrongInput, "DotProductLayer needs at least 2 input tensors");
    }
    for (size_t i = 1; i < num_; i++) {
      if (in_tensors[i].get_dimensions().size() != dims.size()) {
        CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.size(); j++) {
        if (in_tensors[i].get_dimensions()[j] != dims[j]) {
          CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve({num_}, &d_inputs_);
    blobs_buff->reserve(out_tensor.get_dimensions(), &fprop_output_);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void DotProductLayer::initialize() {
  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> pinned_host_buf =
      GeneralBuffer2<CudaHostAllocator>::create();
  pinned_host_buf->reserve({num_}, &h_inputs_);
  pinned_host_buf->allocate();
}

void DotProductLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(float*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  float* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  dot_product_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(d_inputs_.get_ptr(),
                                                                         output, size_, num_);

  CK_CUDA_THROW_(cudaMemcpyAsync((void*)fprop_output_.get_ptr(), (void*)output,
                                 out_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

void DotProductLayer::bprop() {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(float*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  float* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  dot_product_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      output, d_inputs_.get_ptr(), fprop_output_.get_ptr(), size_, num_);
}

}  // namespace HugeCTR
