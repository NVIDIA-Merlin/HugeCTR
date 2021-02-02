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
#include <layers/add_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32
template <typename T>
__global__ void add_kernel(T** inputs, T* output, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    T tmp = 0;
    for (int i = 0; i < num; i++) {
      tmp += inputs[i][tid];
    }
    output[tid] = tmp;
  }
}

template <typename T>
__global__ void add_dgrad_kernel(const T* top_grad, T** dgrads, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size / 2) {
    for (int i = 0; i < num; i++) {
      dgrads[i][tid] = top_grad[tid];
    }
  }
}

template <>
__global__ void add_kernel<__half>(__half** inputs, __half* output, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const __half2** inputs2 = (const __half2**)(inputs);
  __half2* output2 = (__half2*)(output);
  const __half2 zero = __half2half2(__float2half(0.f));

  if (tid < size / 2) {
    __half2 tmp = zero;
    for (int i = 0; i < num; i++) {
      tmp = __hadd2(tmp, inputs2[i][tid]);
    }
    output2[tid] = tmp;
  }
}

template <>
__global__ void add_dgrad_kernel<__half>(const __half* top_grad, __half** dgrads, int size,
                                         int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const __half2* top_grad2 = (const __half2*)(top_grad);
  __half2** dgrads2 = (__half2**)(dgrads);

  if (tid < size) {
    for (int i = 0; i < num; i++) {
      dgrads2[i][tid] = top_grad2[tid];
    }
  }
}

}  // end of namespace

template <typename T>
AddLayer<T>::AddLayer(const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
                      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                      const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      CK_THROW_(Error_t::WrongInput, "AddLayer needs at least 2 input tensors");
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

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void AddLayer<T>::initialize() {
  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> pinned_host_buf =
      GeneralBuffer2<CudaHostAllocator>::create();
  pinned_host_buf->reserve({num_}, &h_inputs_);
  pinned_host_buf->allocate();

  for (size_t i = 0; i < num_; i++) {
    h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
  }

  CK_CUDA_THROW_(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                 num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void AddLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* output = out_tensors_[0].get_ptr();

  dim3 block_size(256, 1, 1);
  dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
  add_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(d_inputs_.get_ptr(), output,
                                                                   size_, num_);
}

template <typename T>
void AddLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  T* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  add_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(output, d_inputs_.get_ptr(),
                                                                       size_, num_);
}

template <>
void AddLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  __half* output = out_tensors_[0].get_ptr();

  dim3 block_size(256, 1, 1);
  dim3 grid_size((size_ / 2 + block_size.x - 1) / block_size.x, 1, 1);
  add_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(d_inputs_.get_ptr(), output,
                                                                   size_, num_);
}

template <>
void AddLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  __half* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ / 2 + blockSize.x - 1) / blockSize.x, 1, 1);
  add_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(output, d_inputs_.get_ptr(),
                                                                       size_, num_);
}

template class AddLayer<float>;
template class AddLayer<__half>;

}  // namespace HugeCTR
