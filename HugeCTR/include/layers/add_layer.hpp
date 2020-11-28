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

#include <layer.hpp>

#include <vector>

namespace HugeCTR {

/**
 * Layer which does element-wise add by input tensors.
 * All the input tensors should have the same shape.
 */
template <typename T>
class AddLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer.
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<T> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

 public:
  /**
   * Ctor of AddLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  AddLayer(const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
           const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
           const std::shared_ptr<GPUResource>& gpu_resource);

  void initialize() override;

  /**
   * AddLayer's foward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * AddLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

 private:
  int size_;
  size_t num_;
  Tensor2<T*> h_inputs_;
  Tensor2<T*> d_inputs_;
};

}  // namespace HugeCTR
