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
 * Layer which
 */
class InteractionLayerHalf : public Layer {
  /*
   * stores the master weight tensors of this layer.
   */
  Tensors2<float> weights_;
  /*
   * stores the weight tensors for compute of this layer.
   */
  Tensors2<__half> weights_half_;

  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<__half> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<__half> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<__half> out_tensors_;

 public:
  /**
   * Ctor of InteractionLayerHalf.
   * @param in_bottom_mlp_tensor the input bottom MLP tensor (batch_size, width)
   * @param in_embeddings the input embeddings (batch_size, n_emb, width)
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  InteractionLayerHalf(Tensor2<__half>& in_bottom_mlp_tensor, Tensor2<__half>& in_embeddings,
                       Tensor2<__half>& out_tensor,
                       const std::shared_ptr<GeneralBuffer2>& blobs_buff,
                       cublasHandle_t cublas_handle, int device_id);
  ~InteractionLayerHalf() override;

  /**
   * Interaction's forward pass to gather data to the output tensor
   * @param stream CUDA stream where the forward propagation is executed
   */
  void fprop(bool is_train, cudaStream_t stream) override;
  /**
   * Interaction's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the forward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  Tensors<__half> internal_tensors_;
};

}  // namespace HugeCTR
