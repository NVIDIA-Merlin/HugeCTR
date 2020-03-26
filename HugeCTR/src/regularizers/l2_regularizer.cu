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

#include "HugeCTR/include/regularizers/l2_regularizer.hpp"

#include "HugeCTR/include/utils.cuh"

#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

void launch_initialize_wgrad_kernel(const float* weight, float* wgrad, int num_elements,
                                    int batch_size, float lambda, int n_sms, cudaStream_t stream) {
  auto op = [lambda, batch_size] __device__(const float in) { return (lambda / batch_size) * in; };
  transform_array<<<n_sms * 4, 512, 0, stream>>>(weight, wgrad, num_elements, op);
}

}  // namespace

L2Regularizer::L2Regularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                             const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                             const int batch_size, const float lambda, cublasHandle_t cublas_handle,
                             const int device_id)
    : Regularizer(weight_buff, wgrad_buff, batch_size, device_id),
      lambda_(lambda),
      cublas_handle_(cublas_handle) {}

void L2Regularizer::do_compute_rterm(const float* weight, float* h_rterm, int num_elements,
                                     cudaStream_t stream) {
  CK_CUBLAS_THROW_(cublasSdot(cublas_handle_, num_elements, weight, 1, weight, 1, h_rterm));
  const float alpha = lambda_ / (get_batch_size() * 2);
  *h_rterm *= alpha;
}

void L2Regularizer::do_initialize_wgrad(const float* weight, float* wgrad, int num_elements,
                                        cudaStream_t stream) {
  launch_initialize_wgrad_kernel(weight, wgrad, num_elements, get_batch_size(), lambda_,
                                 get_n_sms(), stream);
}

}  // namespace HugeCTR
