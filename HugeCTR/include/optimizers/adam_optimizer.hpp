/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
template <typename T>
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
  AdamOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-7, float scaler = 1.f);
  /**
   * Constructor of AdamOptimizer.
   * names of hyper-parameters are the same as in Algorithm 1 of Adam paper (arXiv:1412.6980)
   * @param weight_tensors  a lit of weight tensor in dense layers
   * @param wgrad_ gradient for weights
   * @param gpu_resource the GPU where update kernel is launched
   * @param learning_rate learning rate, alpha in Adam paper
   * @param beta1 beta1 in Adam paper
   * @param beta2 beta2 in Adam paper
   * @param epsilon epsilon in Adam paper
   */
  AdamOptimizer(std::optional<WeightTensors> weight_tensors,
                std::optional<WgradTensors<T>> wgrad_tensors,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-7, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

  std::vector<core23::Tensor> get_opt_state_tensors() override { return {m_tensor_, v_tensor_}; }

 private:
  // named as in Algorithm 1 of Adam paper (arXiv:1412.6980)
  // except that alpha is lr_ in class Optimizer
  Tensor2<T> wgrad_;
  Tensor2<float> m_;
  Tensor2<float> v_;
  std::optional<WgradTensors<T>> wgrad_tensors_;
  core23::Tensor m_tensor_;
  core23::Tensor v_tensor_;
  uint64_t t_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

}  // namespace HugeCTR
