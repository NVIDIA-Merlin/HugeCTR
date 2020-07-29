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

#include "HugeCTR/include/layers/add_layer.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include <prims/linalg/reduce.cuh>
#include <prims/cuda_utils.cuh>

#include <algorithm>
#include <functional>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32

template <typename T>
__global__ void add_dgrad_kernel(const T* top_grad, T** dgrads, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < num; i++) {
      dgrads[i][tid] = top_grad[tid];
    }
  }
}

}  // end of namespace

template <typename T>
AddLayer<T>::AddLayer(const std::vector<std::shared_ptr<Tensor<T>>> in_tensors,
                      const std::shared_ptr<Tensor<T>>& out_tensor, int device_id)
    : Layer(device_id) {
  try {
    CudaDeviceContext context(get_device_id());

    size_ = in_tensors[0]->get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0]->get_dims();
    if (num_ < 2) {
      CK_THROW_(Error_t::WrongInput, "AddLayer needs at least 2 input tensors");
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

    CK_CUDA_THROW_(cudaMallocHost((void**)(&h_inputs_), num_ * sizeof(T*)));
    CK_CUDA_THROW_(cudaMalloc((void**)(&d_inputs_), num_ * sizeof(T*)));

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
AddLayer<T>::~AddLayer() {
  try {
    if (h_inputs_) {
      CK_CUDA_THROW_(cudaFreeHost(h_inputs_));
    }
    if (d_inputs_) {
      CK_CUDA_THROW_(cudaFree(d_inputs_));
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

template <typename T>
void AddLayer<T>::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (int i = 0; i < num_; i++) {
      h_inputs_[i] = in_tensors_[i]->get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_, (void*)h_inputs_, num_ * sizeof(T*),
                                   cudaMemcpyHostToDevice, stream));
    initialized_ = true;
  }
  T* output = out_tensors_[0]->get_ptr();

  MLCommon::LinAlg::reduce(output, d_inputs_, size_, num_, (T)0, false, false, stream, false,
          [] __device__(T in, int i) { return in; });

}

template <typename T>
void AddLayer<T>::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (int i = 0; i < num_; i++) {
      h_inputs_[i] = in_tensors_[i]->get_ptr();
    }
    CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_, (void*)h_inputs_, num_ * sizeof(T*),
                                   cudaMemcpyHostToDevice, stream));
    initialized_ = true;
  }
  T* output = out_tensors_[0]->get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  add_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(output, d_inputs_, size_, num_);
}

template class AddLayer<float>;
template class AddLayer<__half>;

}  // namespace HugeCTR
