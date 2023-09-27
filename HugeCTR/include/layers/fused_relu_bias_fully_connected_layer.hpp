/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <trainable_layer.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief
 * This class implements the fully connected layer.
 */
class FusedReluBiasFullyConnectedLayer : public TrainableLayer<__half> {
  // Optimized cublasGemmEx algorithm selection
  cublasLtMatmulAlgo_t falgo_k_;
  cublasLtMatmulAlgo_t balgo_dRelu_;
  cublasLtMatmulAlgo_t balgo_wgrad_;
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
  cublasLtMatmulDesc_t cublas_op_desc_wgrad_ = NULL;

  cublasLtMatmulPreference_t cublas_preference_ = NULL;
  cublasLtMatmulPreference_t cublas_preference_dRelu_ = NULL;
  cublasLtMatmulPreference_t cublas_preference_wgrad_ = NULL;
  size_t cublaslt_workspace_size_ = 1024 * 1024 * 32;
  void* cublaslt_workspace_;
  void* cublaslt_workspace_dRelu_;
  void* cublaslt_workspace_wgrad_;

  /*
   * stores the weight tensors for compute of this layer.
   */
  // std::vector<TensorPtr<float>> master_weights_; It is inherited from Layer, and named as
  // weights_;

  /*
   * stores the weight tensors for compute of this layer.
   */
  // std::vector<TensorPtr<__half>> weights_;
  std::vector<core23::Tensor> weights_half_;

  /*
   * stores the weight gradient tensors of this layer.
   */
  std::vector<core23::Tensor> weights_grad_;

  /*
   * stores the references to the bottom tensors of this layer.
   */
  core23::Tensor mask_in_tensor_;
  core23::Tensor dRelu_in_tensor_;
  core23::Tensor db_in_tensor_;

  /*
   * stores the references to the top tensors of this layer.
   */
  core23::Tensor mask_out_tensor_;
  core23::Tensor dRelu_out_tensor_;
  core23::Tensor db_out_tensor_;

  /*
   * stores the references to the output tensors of GEMM.
   */
  core23::Tensor identity_tensor_;

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

  /*
   * indicates whether overlap dgrad and wgrad
   */
  bool async_mlp_wgrad_;

  /*
   * determines the kind of fusion pattern.
   * There are two fuse patterns available:
   * (fuse_wb_ == true)  DGRAD + DReLU, WGRAD + BGRAD
   * (fuse_wb_ == false) DGRAD + DReLU + BGRAD, WGRAD
   */
  bool fuse_wb_;

  /*
   * indicates whether there is mask in tensor for Head layer
   */
  bool head_mask_in_;

  bool event_overlap_created_;

  cublasHandle_t cublas_handle_wgrad_;

  /*
   * record the event when starting to compute wgrad
   */
  cudaEvent_t event_overlap_;

  /*
   * record the event when finishing computing wgrad (host, async)
   */
  // cudaEvent_t event_overlap_end_;

  std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;

  core23::Tensor& get_bottom_tensor_fprop(bool is_train) { return this->input_tensors_[0]; }

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
  void initialize_dgrad();
  void initialize_wgrad();

  /*
   * Interfaces for unit tests to debug
   */
  std::vector<core23::Tensor>& get_weights_half_tensor() { return weights_half_; }
  std::vector<core23::Tensor>& get_weights_grad_tensor() { return weights_grad_; }

  /*
   * return the cuda event recording the finish point of wgrad
   */
  // cudaEvent_t& get_event_overlap_end() { return event_overlap_end_; }

  /**
   * This is the constructor of the FullyConnectedLayer.
   * It will check whether the format combination of all tensors is supported or not.
   * Only two kinds of tensor formats are supported:
   * (1) weight, input, output, wgrad are all in row-major.
   * (2) weight, input, output, wgrad are all in column-major.
   * @param train_bottom_tensor_fprop: stores the tensor from bottom layer for forward propagation
   * @param train_bottom_tensor_fprop: stores the tensor from bottom layer for forward propagation
   * @param top_tensor_fprop: stores the tensor to top layer when forward propagation
   * @param top_tensor_bprop: stores the tensor to top layer when backward propagation
   * @param pos: stores the position of this layer: HEAD, BODY, TAIL, ISOLATED.
   */
  FusedReluBiasFullyConnectedLayer(
      const core23::Tensor& train_in_tensor, const core23::Tensor& mask_in_tensor,
      const core23::Tensor& dRelu_in_tensor, const core23::Tensor& db_in_tensor,
      const core23::Tensor& train_out_tensor, const core23::Tensor& mask_out_tensor,
      const core23::Tensor& dRelu_out_tensor, core23::Tensor& db_out_tensor,
      const std::shared_ptr<GPUResource>& gpu_resource, const FcPosition_t& pos,
      const Activation_t& act, const bool& skip_dgrad,
      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>(),
      const bool async_mlp_wgrad = false, const bool head_mask_in = false,
      const bool fuse_wb = false);
  FusedReluBiasFullyConnectedLayer(const FusedReluBiasFullyConnectedLayer&) = delete;
  FusedReluBiasFullyConnectedLayer& operator=(const FusedReluBiasFullyConnectedLayer&);

  ~FusedReluBiasFullyConnectedLayer() {
    try {
      if (event_overlap_created_) {
        CudaDeviceContext context(get_device_id());
        HCTR_LIB_THROW(cudaEventDestroy(event_overlap_));
      }
    } catch (const std::exception& error) {
      HCTR_LOG(INFO, WORLD, "FusedReluBiasFullyConnectedLayer Dtor error:%s", error.what());
    }
  };
};

}  // namespace HugeCTR
