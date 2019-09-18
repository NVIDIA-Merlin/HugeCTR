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
 * Layer that concatenates vectors along slot dimension
 */
class ConcatLayer : public Layer {
 public:
  /**
   * Ctor of ConcatLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param the ID list of slots which are concatenated
   * If it is empty, it is just near-zero-overhead in-place reshape from 3D to 2D.
   * Othewise, the only selected slots are concatenated in newly assigned tensor.
   * @param device_id the id of GPU where this layer belongs
   */
  ConcatLayer(Tensor<float>& in_tensor, Tensor<float>& out_tensor, std::vector<int>& selected,
              int device_id);
  ~ConcatLayer() override;

  /**
   * A method of implementing the forward pass of Concat
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * A method of implementing the forward pass of Concat
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  bool in_place_;
  int n_batch_;
  int n_slot_;
  int vector_length_;
  int n_active_slot_;
  int* slot_mask_;
  int n_sm_;
};

}  // namespace HugeCTR
