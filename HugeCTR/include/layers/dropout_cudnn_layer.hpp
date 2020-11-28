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

#include <cudnn.h>
#include <layer.hpp>

namespace HugeCTR {

/**
 * Dropout layer which selects an arbitrary fraction of inputs to 0
 */
template <typename T>
class DropoutCudnnLayer : public Layer {
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
   * Ctor of DropoutCudnnLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param rate fraction of the inputs set to zero., 0 < rate < 1, default = 0.5
   * @param device_id the id of GPU where this layer belongs
   */
  DropoutCudnnLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                    const std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff, float rate,
                    const std::shared_ptr<GPUResource>& gpu_resource);

  ~DropoutCudnnLayer() override;

  /**
   * A method of implementing the forward pass of Dropout
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * A method of implementing the backward pass of Dropout
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop() override;

  const float* mask() const { return mask_.get_ptr(); }

 private:
  cudnnDropoutDescriptor_t dropout_descriptor_;
  float rate_;
  float scale_;
  void* cudnn_status_;
  Tensor2<float> mask_;
  cudnnTensorDescriptor_t in_out_desc_;
  size_t reserveSpaceSizeInBytes_;
};

}  // namespace HugeCTR
