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

#pragma once

#include "HugeCTR/include/optimizer.hpp"

namespace HugeCTR {

/**
 * Adam optimizer
 */
class AdamOptimizer : public Optimizer {
 public:
  /**
   * Constructor of AdamOptimizer.
   * names of hyper-parameters are the same as in Algorithm 1 of Adam paper (arXiv:1412.6980)
   * @param weight weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param alpha learning rate, alpha in Adam paper
   * @param beta1 beta1 in Adam paper
   * @param beta2 beta2 in Adam paper
   * @param epsilon epsilon in Adam paper
   */
  AdamOptimizer(const std::shared_ptr<GeneralBuffer<float>>& weight,
                const std::shared_ptr<GeneralBuffer<float>>& wgrad, int device_id,
                float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
      : Optimizer(weight, wgrad, device_id, alpha),
        m_(weight->get_num_elements(), device_id),
        v_(weight->get_num_elements(), device_id),
        t_(0),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {
    m_.reset_sync();
    v_.reset_sync();
  }

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update(cudaStream_t stream) override;

 private:
  // named as in Algorithm 1 of Adam paper (arXiv:1412.6980)
  // except that alpha is lr_ in class Optimizer
  GeneralBuffer<float> m_;
  GeneralBuffer<float> v_;
  uint64_t t_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

}  // namespace HugeCTR
