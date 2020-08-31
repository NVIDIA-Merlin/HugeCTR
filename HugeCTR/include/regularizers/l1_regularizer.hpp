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

#include <common.hpp>
#include <memory>
#include <regularizer.hpp>
#include <utils.hpp>

namespace HugeCTR {

/**
 * @brief L1Regularizer
 */
template <typename T>
class L1Regularizer : public Regularizer<T> {
 public:
  /*
   * Constructor of L1Regularizer
   * @param weight_buff GeneralBuffer containing all the layers' weights
   * @param wgrad_buff GeneralBuffer containing all the layers' wgrads
   * @param batch_size Network batch size
   * @param device_id Device to be used
   */
  L1Regularizer(const Tensor2<float>& weight_buff, const Tensor2<T>& wgrad_buff,
                const int batch_size, const float lambda,
                const std::shared_ptr<GPUResource>& gpu_resource);

  /*
   * Destructor of L1Regularizer
   */
  ~L1Regularizer() override {}

 private:
  /*
   * Calculate rterm based on the absolute sum and scale it with lambda / (batch_size)
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_compute_rterm(const float* weight, float* h_rterm, int num_elements) override;
  /*
   * Initialize wgrad with +-(lambda / batch_size)
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  void do_initialize_wgrad(const float* weight, T* wgrad, int num_elements) override;

  const float lambda_;
};

}  // namespace HugeCTR
