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

#include <cpu/layer_cpu.hpp>
#include <vector>

namespace HugeCTR {

/**
 * Layer which does element-wise dot product by input tensors.
 * All the input tensors should have the same shape.
 */
template <typename T>
class ElementwiseMultiplyLayerCPU : public LayerCPU {
 public:
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

  /**
   * Ctor of ElementwiseMultiplyLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  ElementwiseMultiplyLayerCPU(const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff);

  void initialize() override;

  /**
   * ElementwiseMultiplyLayer's foward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * ElementwiseMultiplyLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

 private:
  int size_;
  size_t num_;
  Tensor2<T*> h_inputs_;
  bool initialized_{false};
  Tensor2<T> fprop_output_;
};

}  // namespace HugeCTR
