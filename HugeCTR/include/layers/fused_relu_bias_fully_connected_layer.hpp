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

#include <cublasLt.h>
#include <cublas_v2.h>

#include <functional>
#include <layer.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief
 * This class implements the fully connected layer.
 */
class FusedReluBiasFullyConnectedLayer : public Layer {
  // Optimized cublasGemmEx algorithm selection
  cublasLtMatmulAlgo_t falgo_k_;
  cublasLtMatmulAlgo_t balgo_dRelu_;
  cublasGemmAlgo_t balgo_k_{CUBLAS_GEMM_DEFAULT};
  cublasGemmAlgo_t balgo_x_{CUBLAS_GEMM_DEFAULT};
  cublasGemmAlgo_t balgo_b_{CUBLAS_GEMM_DEFAULT};

  cublasLtMatrixLayout_t cublas_kernel_desc_ = NULL;
  cublasLtMatrixLayout_t cublas_top_desc_ = NULL;
  cublasLtMatrixLayout_t cublas_bottom_desc_ = NULL;
  cublasLtMatrixLayout_t cublas_dRelu_top_desc_ = NULL;
  cublasLtMatrixLayout_t cublas_dRelu_bottom_desc_ = NULL;
  cublasLtMatmulDesc_t cublas_op_desc_ = NULL;
  cublasLtMatmulDesc_t cublas_op_desc_bprop_ = NULL;

  cublasLtMatmulPreference_t cublas_preference_ = NULL;
  cublasLtMatmulPreference_t cublas_preference_dRelu_ = NULL;
  size_t cublaslt_workspace_size_ = 1024 * 1024 * 8;
  void* cublaslt_workspace_;
  void* cublaslt_workspace_dRelu_;

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
   * stores the references to the bottom tensors of this layer.
   */
  Tensor2<__half> train_in_tensor_;
  Tensor2<__half> mask_in_tensor_;
  Tensor2<__half> dRelu_in_tensor_;
  Tensor2<__half> db_in_tensor_;
  Tensor2<int> mask_in_tensor_temp_;

  /*
   * stores the references to the top tensors of this layer.
   */
  Tensor2<__half> train_out_tensor_;
  Tensor2<__half> mask_out_tensor_;
  Tensor2<__half> dRelu_out_tensor_;
  Tensor2<__half> db_out_tensor_;

  /*
   * stores the references to the output tensors of GEMM.
   */
  Tensor2<__half> identity_tensor_;

  /*
   * stores the references to the intermediate bias grad tensors of this layer.
   */
  Tensor2<float> bias_grad_tensor_;

  void* bprop_fusion_;

  /*
   * stores the position of this layer in the network
   */
  FcPosition_t pos_;

  /*
   * stores the activation function of this layer
   */
  Activation_t act_;

  /*
   * skip the computation of dgrad or not
   */
  bool skip_dgrad_;

  std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;

  Tensor2<__half>& get_bottom_tensor_fprop(bool is_train) { return train_in_tensor_; }

  Tensor2<__half>& get_bottom_tensor_bprop(bool is_train) { return mask_in_tensor_; }

 public:
  /**
   * forward pass
   */
  void fprop(bool is_train) final;
  /**
   * backward pass
   */
  void bprop() final;
  /*
   * algorithm search for cublasGemmEx
   */
  void search_algorithm() final;
  void initialize() final;
  void initialize_bprop();

  /**
   * This is the constructor of the FullyConnectedLayer.
   * It will check whether the format combination of all tensors is supported or not.
   * Only two kinds of tensor formats are supported:
   * (1) weight, input, output, wgrad are all in row-major.
   * (2) weight, input, output, wgrad are all in column-major.
   * @param weight_buff: stores the weight tensor
   * @param wgrad_buff: stores the gradient values of the weight calculated in backward pass
   * @param train_bottom_tensor_fprop: stores the tensor from bottom layer for forward propagation
   * @param train_bottom_tensor_fprop: stores the tensor from bottom layer for forward propagation
   * @param top_tensor_fprop: stores the tensor to top layer when forward propagation
   * @param top_tensor_bprop: stores the tensor to top layer when backward propagation
   * @param pos: stores the position of this layer: HEAD, BODY, TAIL, ISOLATED.
   */
  FusedReluBiasFullyConnectedLayer(
      const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
      const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
      const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
      const Tensor2<__half>& train_in_tensor, const Tensor2<__half>& mask_in_tensor,
      const Tensor2<__half>& dRelu_in_tensor, const Tensor2<__half>& db_in_tensor,
      const Tensor2<__half>& train_out_tensor, const Tensor2<__half>& mask_out_tensor,
      const Tensor2<__half>& dRelu_out_tensor, Tensor2<__half>& db_out_tensor,
      const std::shared_ptr<GPUResource>& gpu_resource, const FcPosition_t& pos,
      const Activation_t& act, const bool& skip_dgrad,
      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  FusedReluBiasFullyConnectedLayer(const FusedReluBiasFullyConnectedLayer&) = delete;
  FusedReluBiasFullyConnectedLayer& operator=(const FusedReluBiasFullyConnectedLayer&);
};
}  // namespace HugeCTR
