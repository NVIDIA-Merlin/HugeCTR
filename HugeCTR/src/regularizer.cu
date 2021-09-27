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

#include <regularizer.hpp>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
Regularizer<T>::Regularizer(const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
                            const int batch_size, const std::shared_ptr<GPUResource>& gpu_resource)
    : overlapped_(false),
      weight_buff_(weight_buff),
      wgrad_buff_(wgrad_buff),
      batch_size_(batch_size),
      gpu_resource_(gpu_resource) {
  CudaDeviceContext context(get_device_id());
  cudaEventCreateWithFlags(&fork_event_, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&join_event_, cudaEventDisableTiming);
  cudaStreamCreateWithFlags(&reg_stream_, cudaStreamNonBlocking);
}

template <typename T>
Regularizer<T>::~Regularizer() {
  cudaEventDestroy(fork_event_);
  cudaEventDestroy(join_event_);
  cudaStreamDestroy(reg_stream_);
}

template <typename T>
void Regularizer<T>::compute_rterm() {
  CudaDeviceContext context(get_device_id());

  const float* weight = weight_buff_.get_ptr();
  do_compute_rterm(weight, &h_rterm_, weight_buff_.get_num_elements());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void Regularizer<T>::initialize_wgrad() {
  CudaDeviceContext context(get_device_id());

  const float* weight = weight_buff_.get_ptr();
  T* wgrad = wgrad_buff_.get_ptr();
  do_initialize_wgrad(weight, wgrad, weight_buff_.get_num_elements(), get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void Regularizer<T>::initialize_wgrad_async() {
  CudaDeviceContext context(get_device_id());
  const float* weight = weight_buff_.get_ptr();
  T* wgrad = wgrad_buff_.get_ptr();
  CK_CUDA_THROW_(cudaEventRecord(fork_event_, get_gpu().get_stream()));
  CK_CUDA_THROW_(cudaStreamWaitEvent(reg_stream_, fork_event_));
  do_initialize_wgrad(weight, wgrad, weight_buff_.get_num_elements(), reg_stream_);
  CK_CUDA_THROW_(cudaEventRecord(join_event_, reg_stream_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void Regularizer<T>::join_initialize_wgrad() {
  CK_CUDA_THROW_(cudaStreamWaitEvent(get_gpu().get_stream(), join_event_));
}

template class Regularizer<float>;
template class Regularizer<__half>;
}  // namespace HugeCTR
