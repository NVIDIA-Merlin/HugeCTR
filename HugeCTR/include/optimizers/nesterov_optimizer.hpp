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

#include <optimizer.hpp>

namespace HugeCTR {

/**
 * SGD optimizer with Nesterov Momentum
 */
class NesterovOptimizer : public Optimizer {
 public:
  /**
   * Constructor of NesterovOptimizer.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param momentum_factor the momentum factor
   */
  NesterovOptimizer(const Tensor2<float>& weight_main, const Tensor2<float>& fp32_wgrad,
                    const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
                    const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                    const std::shared_ptr<BufferBlock2<__half>>& opt_buf_half,
                    const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate,
                    float momentum_factor, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

 private:
  Tensor2<float> fp32_accum_;   // accumulation
  Tensor2<__half> fp16_accum_;  // accumulation
  const float mu_;              // momentum factor
};

}  // namespace HugeCTR
