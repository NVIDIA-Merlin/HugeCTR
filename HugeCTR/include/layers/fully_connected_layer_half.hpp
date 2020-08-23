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

#include <functional>
#include <layer.hpp>
#include <vector>
#include "cublas_v2.h"

namespace HugeCTR {
/**
 * @brief
 * This class implements the fully connected layer.
 */
class FullyConnectedLayerHalf : public Layer {
  const cublasHandle_t cublas_handle_;
  // Optimized cublasGemmEx algorithm selection
  cublasGemmAlgo_t falgo_b_;
  cublasGemmAlgo_t falgo_k_;
  cublasGemmAlgo_t balgo_b_;
  cublasGemmAlgo_t balgo_k_;
  cublasGemmAlgo_t balgo_x_;

  /*
   * stores the weight tensors for compute of this layer.
   */
  // std::vector<TensorPtr<float>> master_weights_; It is inherited from Layer, and named as
  // weights_;

  /*
   * stores the weight tensors for compute of this layer.
   */
  // std::vector<TensorPtr<__half>> weights_;
  Tensors2<__half> weights_half_;

  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<__half> weights_grad_;

  /*
   * stores the references to the input tensors of this layer.
   */
  Tensor2<__half> train_bottom_tensor_;
  Tensor2<__half> evaluate_bottom_tensor_;

  /*
   * stores the references to the output tensors of this layer.
   */
  Tensor2<__half> top_tensor_;

  /*
   * stores the references to the output tensors of GEMM.
   */
  Tensor2<__half> identity_tensor_;

  /*
 * initializers for this layer.
 */
  std::unique_ptr<DataSimulator<float>> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator<float>> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator<float>> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator<float>> get_default_initializer(const int index) override;

  Tensor2<__half>& get_bottom_tensor(bool is_train) {
    if (is_train) {
      return train_bottom_tensor_;
    } else {
      return evaluate_bottom_tensor_;
    }
  }

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train, cudaStream_t stream) final;
  /**
   * backward pass
   */
  void bprop(cudaStream_t stream) final;
  /*
   * initialize for cublasGemmEx
   */
  void initialize() final;
  /*
   * algorithm search for cublasGemmEx
   */
  void search_algorithm() final;
  /**
   * This is the constructor of the FullyConnectedLayer.
   * It will check whether the format combination of all tensors is supported or not.
   * Only two kinds of tensor formats are supported:
   * (1) weight, input, output, wgrad are all in row-major.
   * (2) weight, input, output, wgrad are all in column-major.
   * @param weight_buff: stores the weight tensor
   * @param wgrad_buff: stores the gradient values of the weight calculated in backward pass
   * @param bottom_tensor: stores the tensor from bottom layer
   * @param top_tensor: stores the tensor to top layer
   * @param tensor_format: specifies the format of the weight tensor, either HW (row major) or WH
   * (col-major)
   */
  FullyConnectedLayerHalf(
      const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
      const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
      const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
      const Tensor2<__half>& train_bottom_tensor, const Tensor2<__half>& evaluate_bottom_tensor,
      const Tensor2<__half>& top_tensor, cublasHandle_t const& cublas_handle, int device_id,
      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  FullyConnectedLayerHalf(const FullyConnectedLayerHalf&) = delete;
  FullyConnectedLayerHalf& operator=(const FullyConnectedLayerHalf&);
};
}  // namespace HugeCTR
