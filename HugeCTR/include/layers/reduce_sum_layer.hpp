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
#include <vector>

namespace HugeCTR {

/**
 * Layer which does reduce-sum operation by input tensor.
 * The reduced axis(dimension) can be selected. The output
 * tensor will keep the reduced dimension.
 */
template <typename T>
class ReduceSumLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<T> weights_;
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
   * Ctor of ReduceSumLayer.
   * @param in_tensor the input tensor, could be 2D or 3D
   * @param out_tensor the resulting output tensor
   * @param axis the reduced dimension, could be 0,1,2
   * @param device_id the id of GPU where this layer belongs
   */
  ReduceSumLayer(const Tensor2<T>& in_tensors, Tensor2<T>& out_tensor,
                 const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff, int axis,
                 const std::shared_ptr<GPUResource>& gpu_resource);
  ~ReduceSumLayer(){};

  /**
   * ReduceSumLayer's forward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * ReduceSumLayer's backward propagation
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop() override;

 private:
  int axis_;
};

}  // namespace HugeCTR
