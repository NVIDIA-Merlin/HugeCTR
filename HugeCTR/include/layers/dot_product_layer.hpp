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

#include "HugeCTR/include/layer.hpp"

#include <vector>

namespace HugeCTR {

/**
 * Layer which does element-wise dot product by input tensors.
 * All the input tensors should have the same shape.
 */
class DotProductLayer : public Layer {
 public:
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<float>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<float>>> out_tensors_;


  /**
   * Ctor of DotProductLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  DotProductLayer(const std::vector<std::shared_ptr<Tensor<float>>>& in_tensors,
  const std::shared_ptr<Tensor<float>>& out_tensor, int device_id);
  ~DotProductLayer();

  /**
   * DotProductLayer's foward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * DotProductLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  int size_;
  int num_;
  float** h_inputs_ = NULL;
  float** d_inputs_ = NULL;
  bool initialized_{false};
  float* fprop_output_ = NULL;
};

}  // namespace HugeCTR
