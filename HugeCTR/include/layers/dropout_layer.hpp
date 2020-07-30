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

#include <curand.h>

namespace HugeCTR {

/**
 * Dropout layer which selects an arbitrary fraction of inputs to 0
 */
template <typename T>
class DropoutLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer.
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors<T> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<T>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<T>>> out_tensors_;

 public:
  /**
   * Ctor of DropoutLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param rate fraction of the inputs set to zero., 0 < rate < 1, default = 0.5
   * @param device_id the id of GPU where this layer belongs
   */
  DropoutLayer(const std::shared_ptr<Tensor<T>>& in_tensor,
               const std::shared_ptr<Tensor<T>>& out_tensor, float rate,
               const curandGenerator_t& curand_generator, int device_id);

  ~DropoutLayer() override;

  /**
   * A method of implementing the forward pass of Dropout
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * A method of implementing the backward pass of Dropout
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop(cudaStream_t stream) override;
  /*
   * Inference pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  void inference(cudaStream_t stream) override;

  const float* mask() const { return mask_; }

  void prop_common(const T* in, T* out, cudaStream_t stream);

 private:
  int64_t get_seed() const;

  float rate_;
  float scale_;
  float* mask_;
  curandGenerator_t curand_generator_;
  int n_sms_;
};

}  // namespace HugeCTR
