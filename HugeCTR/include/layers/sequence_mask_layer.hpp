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

#include <layer.hpp>

namespace HugeCTR {

/**
 * SequenceMaskLayerfunction as a derived class of Layer
 */
template <typename T>
class SequenceMaskLayer : public Layer {
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensor2<T> in_tensor_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensor2<T> out_tensor_;

 public:
  /**
   * Ctor of SequenceMaskLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this layer belongs
   */
  SequenceMaskLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, int max_sequence_len,
                    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                    const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * A method of implementing the forward pass of SequenceMaskLayer
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * A method of implementing the backward pass of SequenceMaskLayer
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop() override;

  /*
   * stores the max sequence length.
   */
  int max_sequence_len_;
};

}  // namespace HugeCTR
