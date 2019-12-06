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
 * Layer which reshapes a 3D/2D input tensor to 2D output tensor,
 * e.g., (batch_size, n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * e.g., (batch_size * n_slots, vector_size) to (batch_size, n_slots * vector_size),
 * If the input tensor is 3D, you can choose which slots participate by calling the different Ctor
 */
class ReshapeLayer : public Layer {
 public:
  /**
   * General Purpose Ctor of ReshapeLayer.
   * @param in_tensor the input tensor
   * @param out_tensor a double pointer to the resulting output tensor
   * @param leading_dim must be a multiple of the innermost dimesion
   * e.g., leading_dim % vector_size == 0
   * and it must be able to divide the total number of elements in in_tensor
   * e.g., batch_size * n_slots * vector_size % leading_dim == 0
   * @param device_id the id of GPU where this layer belongs
   */
  ReshapeLayer(Tensor<float>& in_tensor, Tensor<float>** out_tensor, int leading_dim,
               int device_id);
  /**
   * Specialized Ctor of ReshapeLayer which assumes the 3D input tensor
   * @param in_tensor the input tensor
   * @param out_tensor a double pointer to the resulting output tensor
   * @param the ID list of slots which are concatenated
   * If it is empty, it is just near-zero-overhead in-place reshape from 3D to 2D.
   * Othewise, the only selected slots are concatenated in newly assigned tensor.
   * @param device_id the id of GPU where this layer belongs
   */
  ReshapeLayer(Tensor<float>& in_tensor, Tensor<float>** out_tensor,
               GeneralBuffer<float>& blobs_buff, 
               std::vector<int>& selected, int device_id);
  ~ReshapeLayer() override;

  /**
   * A method of implementing the forward pass of Reshape
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * A method of implementing the forward pass of Reshape
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  void prop_common(bool forward, cudaStream_t stream);

  bool in_place_;
  int batch_size_;
  int n_slot_;
  int vector_length_;
  int n_active_slot_;
  int* selected_;
  int n_sms_;
};

}  // namespace HugeCTR
