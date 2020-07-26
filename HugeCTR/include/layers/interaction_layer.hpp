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

#include <vector>

namespace HugeCTR {

/**
 * Layer which
 */
template <typename T>
class InteractionLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors<T> weights_;
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
   * Ctor of InteractionLayer.
   * @param in_bottom_mlp_tensor the input bottom MLP tensor (batch_size, width)
   * @param in_embeddings the input embeddings (batch_size, n_emb, width)
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  InteractionLayer(std::shared_ptr<Tensor<T>> in_bottom_mlp_tensor,
                   std::shared_ptr<Tensor<T>> in_embeddings, std::shared_ptr<Tensor<T>>& out_tensor,
                   const std::shared_ptr<GeneralBuffer<T>>& blobs_buff,
                   cublasHandle_t cublas_handle, bool use_mixed_precision, int device_id);
  ~InteractionLayer() override;

  /**
   * Interaction's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * Interaction's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  cublasHandle_t cublas_handle_;
  bool use_mixed_precision_;
  int n_sms_;

  Tensors<T> internal_tensors_;
};

}  // namespace HugeCTR
