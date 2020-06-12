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

#pragma once

#include "HugeCTR/include/regularizer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/utils.hpp"

#include <memory>

namespace HugeCTR {

/**
 * @brief NoRegularizer
 */
template<typename T>
class NoRegularizer : public Regularizer<T> {
 public:
  /*
   * Constructor of NoRegularizer
   * @param weight_buff GeneralBuffer containing all the layers' weights
   * @param wgrad_buff GeneralBuffer containing all the layers' wgrads
   * @param batch_size Network batch size
   * @param device_id Device to be used
   */
  NoRegularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                const std::shared_ptr<GeneralBuffer<T>>& wgrad_buff, const int batch_size,
                const int device_id);

  /*
   * Destructor of NoRegularizer
   */
  ~NoRegularizer() override {}

 private:
  /*
   * rterm is zero
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_compute_rterm(const float* weight, float* h_rterm, int num_elements,
                        cudaStream_t stream) override;
  /*
   * Initialize wgrad with zeros
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_initialize_wgrad(const float* weight, T* wgrad, int num_elements,
                           cudaStream_t stream) override;
};

}  // namespace HugeCTR
