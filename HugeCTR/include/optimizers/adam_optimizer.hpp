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

#pragma once

#include <general_buffer2.hpp>
#include <optimizer.hpp>

namespace HugeCTR {

/**
 * Adam optimizer
 */
class AdamOptimizer : public Optimizer {
 public:
  /**
   * Constructor of AdamOptimizer.
   * names of hyper-parameters are the same as in Algorithm 1 of Adam paper (arXiv:1412.6980)
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate, alpha in Adam paper
   * @param beta1 beta1 in Adam paper
   * @param beta2 beta2 in Adam paper
   * @param epsilon epsilon in Adam paper
   */
  AdamOptimizer(const Tensor2<float>& weight_main, const Tensor2<float>& fp32_wgrad,
                const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
                const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& buff,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-7, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

 private:
  // named as in Algorithm 1 of Adam paper (arXiv:1412.6980)
  // except that alpha is lr_ in class Optimizer
  Tensor2<float> fp32_m_;
  Tensor2<float> fp32_v_;
  Tensor2<__half> fp16_m_;
  Tensor2<__half> fp16_v_;
  uint64_t t_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

}  // namespace HugeCTR
