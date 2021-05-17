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

#include <general_buffer2.hpp>
#include <optimizer.hpp>

namespace HugeCTR {

/**
 * AdaGrad optimizer
 */
template <typename T>
class AdaGradOptimizer : public Optimizer {
 public:
  /**
   * Constructor of AdaGradOptimizer.
   * names of hyper-parameters are the same as in AdaGrad paper (https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate, alpha in Adam paper
   * @param beta1 beta1 in Adam paper
   * @param beta2 beta2 in Adam paper
   * @param epsilon epsilon in Adam paper
   */
  AdaGradOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                const std::shared_ptr<BufferBlock2<T>>& opt_buf,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float initial_accu_value = 0., float epsilon = 1e-7, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

 private:
  Tensor2<T> wgrad_;
  Tensor2<T> accum_;
  
  float initial_accumulator_value_;
  const float epsilon_;
};

}  // namespace HugeCTR