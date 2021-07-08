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
 * Layer which does element-wise product by input tensor X and weight W.
 * The input tensor X has dimention: [batch_size, slot_num], while
 * the input weight W has dimention: [slot_num, embedding_vec_size].
 * The WeightMultiplyLayer will broadcast the value of W to "batch_size" dim
 * and broadcast the value of X to embedding_vec_size dim automatically
 * when doing element-wise product with X. So, the output tensor has
 * the dimention: [batch_size, slot_num*embedding_vec_size].
 */
template <typename T>
class WeightMultiplyLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<float> master_weights_;
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
   * Ctor of WeightMultiplyLayer.
   * @param in_tensor the input tensor
   * @param out_tensor the resulting output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  WeightMultiplyLayer(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                const std::shared_ptr<BufferBlock2<T>>& weight_buff,
                const std::shared_ptr<BufferBlock2<T>>& wgrad_buff,
                const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blob_buff,
                const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                const std::vector<size_t>& weight_dims,
                const std::shared_ptr<GPUResource>& gpu_resource,
                std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());

  ~WeightMultiplyLayer() override{};

  /**
   * WeightMultiplyLayer's foward propagation to do element-wise production
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * WeightMultiplyLayer's backward propagation
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

 private:
  void reserve_master_weight_tensor(const std::shared_ptr<BufferBlock2<float>>& master_weight_buff,
                                    const std::vector<size_t>& weight_dims);
  std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;

  size_t batch_size_;
  size_t slot_num_;
  size_t embedding_vec_size_;
  Tensor2<T> wgrad_tmp_trans_;
};

}  // namespace HugeCTR
