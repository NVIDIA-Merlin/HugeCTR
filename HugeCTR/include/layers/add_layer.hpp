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
 * Layer which does element-wise add by input tensors. 
 * All the input tensors should have the same shape.
 */
class AddLayer : public Layer {
 public:
  /**
   * Ctor of AddLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  AddLayer(const std::vector<std::shared_ptr<Tensor<float>>>& in_tensors,
          const std::shared_ptr<Tensor<float>>& out_tensor,
          int device_id);
  ~AddLayer();

  /**
   * AddLayer's foward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * AddLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
   int size_;
   int num_;
   float ** h_inputs_ = NULL;
   float ** d_inputs_ = NULL;
   bool initialized_{false};
};

}  // namespace HugeCTR
