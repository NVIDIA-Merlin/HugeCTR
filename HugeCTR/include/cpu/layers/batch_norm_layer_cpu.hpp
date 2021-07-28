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

namespace HugeCTR {

/**
 * BatchNorm layer based on cuDNN
 */
template <typename T>
class BatchNormLayerCPU : public LayerCPU {
  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer, and named as weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<float> wgrad_;
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
   * BatchNorm parameters
   */
  struct Params {
    double factor; /**<  moving average computation factor*/
    double eps;    /**< small value to avoid divide-by-zero error*/
  };

  /**
   * Ctor of BatchNormLayer.
   * @param weight_buff weight buffer for internal gamma/beta tensors
   * @param wgrad_buff gradient buffer for internal gamma/beta tensors
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param params BatchNorm parameters
   * @param cudnn_handle cuDNN handle created externally
   * @param device_id the id of GPU where this layer belongs
   */
  BatchNormLayerCPU(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                    const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                    const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blob_buff,
                    const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                    const Params& params);
  ~BatchNormLayerCPU() override;

  void initialize() override;

  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;

  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

 private:
  const Params params_;

  // these four pointers are just for convenience
  // they are deleted by Layer d'tor through the other pointer aliases: weight_ and wgrad_
  Tensor2<float> gamma_;
  Tensor2<float> beta_;
  Tensor2<float> gamma_grad_;
  Tensor2<float> beta_grad_;
};

}  // namespace HugeCTR
