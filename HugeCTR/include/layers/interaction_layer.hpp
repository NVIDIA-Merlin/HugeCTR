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
template <typename T>
class InteractionLayer : public Layer {
  bool enable_tf32_compute_;
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

  bool use_mixed_precision_;

  Tensors2<T> internal_tensors_;

  Tensors2<T>& get_in_tensors(bool is_train) { return in_tensors_; }

 public:
  /**
   * Ctor of InteractionLayer.
   * @param in_bottom_mlp_tensor the input bottom MLP tensor (batch_size, width)
   * @param in_embeddings the input embeddings (batch_size, n_emb, width)
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  InteractionLayer(const Tensor2<T>& in_bottom_mlp_tensor, const Tensor2<T>& in_embeddings,
                   Tensor2<T>& out_tensor,
                   const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                   const std::shared_ptr<GPUResource>& gpu_resource,
                   bool use_mixed_precision, bool enable_tf32_compute);
  ~InteractionLayer() override;

  /**
   * Interaction's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * Interaction's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;
};

}  // namespace HugeCTR
