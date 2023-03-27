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

#include <optimizer.hpp>

namespace HugeCTR {

/**
 * SGD optimizer with Momentum
 */
template <typename T>
class MomentumSGDOptimizer : public Optimizer {
 public:
  /**
   * Constructor of MomentumSGD.
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param momentum_factor momentum factor
   */
  MomentumSGDOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                       const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                       const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate,
                       float momentum_factor, float scaler = 1.f);
  /**
   * Constructor of MomentumSGD.
   * @param weight_tensors a list of dense layer weight tensors
   * @param wgrad gradient for weight_tensors
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param momentum_factor momentum factor
   */
  MomentumSGDOptimizer(std::optional<WeightTensors> weight_tensors,
                       std::optional<WgradTensors<T>> wgrad_tensors,
                       const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate,
                       float momentum_factor, float scaler = 1.f);

  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update() override;

  std::vector<core23::Tensor> get_opt_state_tensors() override { return {momentum_tensor_}; }

 private:
  Tensor2<T> wgrad_;
  Tensor2<float> momentum_;
  std::optional<WgradTensors<T>> wgrad_tensors_;
  core23::Tensor momentum_tensor_;
  const float momentum_factor_;
};

}  // namespace HugeCTR
