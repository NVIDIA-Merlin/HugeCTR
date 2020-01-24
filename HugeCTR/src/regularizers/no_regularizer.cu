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

#include "HugeCTR/include/regularizers/no_regularizer.hpp"

#include "HugeCTR/include/utils.cuh"

#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

NoRegularizer::NoRegularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                             const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                             const int batch_size,
                             const int device_id)
    : Regularizer(weight_buff, wgrad_buff, batch_size, device_id) {
}

void NoRegularizer::do_compute_rterm(const float* weight, float* rterm,
                                     int num_elements,
                                     cudaStream_t stream) {
  *rterm = 0.0f;
}

void NoRegularizer::do_initialize_wgrad(const float* weight, float* wgrad,
                                      int num_elements,
                                      cudaStream_t stream) {
  int n_blocks = get_n_sms() * 4;
  int block_size = 512;
  initialize_array<<<n_blocks, block_size, 0, stream>>>(wgrad, num_elements, 0.0f);
}

}  // namespace HugeCTR
