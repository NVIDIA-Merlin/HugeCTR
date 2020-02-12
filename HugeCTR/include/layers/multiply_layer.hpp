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

#include "HugeCTR/include/layer.hpp"

#include <vector>

namespace HugeCTR {

/**
 * Layer which does element-wise product by input vector X and weight W. 
 * The input tensor X has dimention: [batch_size, vector_length], while 
 * the input weight W has dimention: [1, vector_length]. The MultiplyLayer
 * can broadcast the value of W from vector_length dim to batch_size dim
 * automatically when doing element-wise product with X and W. So, the output
 * tensor has the same dimention size with the input tensor X.
 */
class MultiplyLayer : public Layer {
 public:
  /**
   * Ctor of MultiplyLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  MultiplyLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                const std::shared_ptr<Tensor<float>>& in_tensor,
                const std::shared_ptr<Tensor<float>>& out_tensor,
                int device_id);
  ~MultiplyLayer() override {};

  /**
   * MultiplyLayer's foward propagation to do element-wise production
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * MultiplyLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:

};

}  // namespace HugeCTR
