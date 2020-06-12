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

#include "HugeCTR/include/regularizer.hpp"

#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template<typename T>
Regularizer<T>::Regularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                         const std::shared_ptr<GeneralBuffer<T>>& wgrad_buff,
                         const int batch_size, const int device_id)
    : weight_buff_(weight_buff),
      wgrad_buff_(wgrad_buff),
      batch_size_(batch_size),
      device_id_(device_id),
      n_sms_(0),
      h_rterm_(new float) {
  CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sms_, cudaDevAttrMultiProcessorCount, device_id_));
}

template<typename T>
void Regularizer<T>::compute_rterm(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const float* weight = weight_buff_->get_ptr_with_offset(0);
  do_compute_rterm(weight, h_rterm_.get(), weight_buff_->get_num_elements(), stream);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template<typename T>
void Regularizer<T>::initialize_wgrad(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const float* weight = weight_buff_->get_ptr_with_offset(0);
  T* wgrad = wgrad_buff_->get_ptr_with_offset(0);
  do_initialize_wgrad(weight, wgrad, weight_buff_->get_num_elements(), stream);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class Regularizer<float>;
template class Regularizer<__half>;
}  // namespace HugeCTR
