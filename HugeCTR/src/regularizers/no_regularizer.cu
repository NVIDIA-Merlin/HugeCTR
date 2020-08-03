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

#include <regularizers/no_regularizer.hpp>
#include <utils.cuh>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {
template <typename T>
NoRegularizer<T>::NoRegularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                                const std::shared_ptr<GeneralBuffer<T>>& wgrad_buff,
                                const int batch_size, const int device_id)
    : Regularizer<T>(weight_buff, wgrad_buff, batch_size, device_id) {}

template <typename T>
void NoRegularizer<T>::do_compute_rterm(const float* weight, float* rterm, int num_elements,
                                        cudaStream_t stream) {
  *rterm = 0.0f;
}

template <typename T>
void NoRegularizer<T>::do_initialize_wgrad(const float* weight, T* wgrad, int num_elements,
                                           cudaStream_t stream) {
  int n_blocks = Regularizer<T>::get_n_sms() * 4;
  int block_size = 512;
  initialize_array<<<n_blocks, block_size, 0, stream>>>(wgrad, num_elements, T(0.0f));
}

template class NoRegularizer<__half>;
template class NoRegularizer<float>;

}  // namespace HugeCTR
