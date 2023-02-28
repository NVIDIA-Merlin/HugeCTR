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
 * Ftrl optimizer
 */
template <typename T>
class FtrlOptimizer : public Optimizer {
 public:
  /**
   * Constructor of FtrlOptimizer.
   * names of hyper-parameters are the same as in FTRL paper "Ad Click Prediction: a View from the
   * Trenches"
   * @param weight_main weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate alpha in FTRL paper
   * @param beta beta in FTRL paper
   * @param lambda1 lambda1 in FTRL paper
   * @param lambda2 lambda2 in FTRL paper
   */
  FtrlOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float beta = 0.0f, float lambda1 = 0.0f, float lambda2 = 0.0f, float scaler = 1.f);
  /**
   * Constructor of FtrlOptimizer.
   * names of hyper-parameters are the same as in FTRL paper "Ad Click Prediction: a View from the
   * Trenches"
   * @param weight_tensors a list of dense layer weight tensors
   * @param wgrad_tensors gradient for weight_tensors
   * @param gpu_resource the GPU where update kernel is launched
   * @param learning_rate learning rate alpha in FTRL paper
   * @param beta beta in FTRL paper
   * @param lambda1 lambda1 in FTRL paper
   * @param lambda2 lambda2 in FTRL paper
   */
  FtrlOptimizer(std::vector<core23::Tensor> weight_tensors,
                std::vector<core23::Tensor> wgrad_tensors,
                const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate = 0.001,
                float beta = 0.0f, float lambda1 = 0.0f, float lambda2 = 0.0f, float scaler = 1.f);
  /*Initialization:
  ```python
  n = 0
  sigma = 0
  z = 0*/
  void initialize() override;

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel

  ```
  Update rule for one variable `w`:
  ```python
  prev_n = n
  n = n + g ** 2
  sigma = (sqrt(n) - sqrt(prev_n)) / lr
  z = z + g - sigma * w
  if abs(z) < lambda_1:
    w = 0
  else:
    w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / lr + lambda_2)
  ```
   */
  void update() override;

 private:
  // named as in ftrl paper Ad Click Prediction: a View from the Trenches
  // except alpha is lr_ in class Optimizer
  Tensor2<T> wgrad_;
  Tensor2<float> z_;
  Tensor2<float> n_;

  std::optional<WgradTensors<T>> wgrad_tensors_;
  core23::Tensor z_tensor_;
  core23::Tensor n_tensor_;
  // uint64_t t_;
  const float beta_;
  const float lambda1_;
  const float lambda2_;
};

}  // namespace HugeCTR
