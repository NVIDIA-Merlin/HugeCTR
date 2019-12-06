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
 * Layer which merges the multiple 2D input tensors to a single 2D output tensor.
 * The input tensors and the resulting output tensor must have the same dimensionallity.
 * Only the innermost dimension is expanded by concatenating those of the input tensors.
 * e.g., 3X(batch_size, n_slots * vector_size) to (batch_size, 3 * n_slots * vector_size),
 * e.g., (batch_size, a * vector_size) + (batch_size, b * vector_size)
 *       to (batch_size, (a + b) * vector_size)
 */
class ConcatLayer : public Layer {
 public:
  /**
   * Ctor of ConcatLayer.
   * @param in_tensors the vector of the input tensors
   * @param out_tensor a double pointer to the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  ConcatLayer(std::vector<Tensor<float>*>& in_tensors, Tensor<float>** out_tensor,
               GeneralBuffer<float>& blobs_buff, int device_id);
  ~ConcatLayer() override {};

  /**
   * Concat's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * Concat's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

  template <typename T>
  struct InParam {
    T* in;
    const int in_w;
  };

 private:
  void prop_common(bool forward, cudaStream_t stream);
  std::vector<InParam<float>> set_in_params(int n);
  template <typename... Args>
  void kernel_launch(bool forward, cudaStream_t stream, Args&... args);

  int n_sms_;
};

}  // namespace HugeCTR
