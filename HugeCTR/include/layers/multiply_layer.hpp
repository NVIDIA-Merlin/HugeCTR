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
 * Layer which does element-wise product by input tensor X and weight W.
 * The input tensor X has dimention: [batch_size, slot_num], while
 * the input weight W has dimention: [slot_num, embedding_vec_size].
 * The MultiplyLayer will broadcast the value of W to "batch_size" dim
 * and broadcast the value of X to embedding_vec_size dim automatically
 * when doing element-wise product with X. So, the output tensor has
 * the dimention: [batch_size, slot_num*embedding_vec_size].
 */
class MultiplyLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors<float> weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors<float> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<float>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<float>>> out_tensors_;

 public:
  /**
   * Ctor of MultiplyLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  MultiplyLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                const std::shared_ptr<GeneralBuffer<float>>& blob_buff,
                const std::shared_ptr<Tensor<float>>& in_tensor,
                std::shared_ptr<Tensor<float>>& out_tensor, const std::vector<size_t>& weight_dims,
                int device_id);
  ~MultiplyLayer() override{};

  /**
   * MultiplyLayer's foward propagation to do element-wise production
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * MultiplyLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

 private:
  /**
   * Use Gaussian initialization.
   */
  std::vector<float> get_initializer() override;

  size_t batch_size_;
  size_t slot_num_;
  size_t embedding_vec_size_;
  std::shared_ptr<GeneralBuffer<float>> internal_buff_;
  std::unique_ptr<Tensor<float>> wgrad_tmp_trans_;
};

}  // namespace HugeCTR
