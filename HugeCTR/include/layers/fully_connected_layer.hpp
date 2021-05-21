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

#include <cublas_v2.h>

#include <functional>
#include <layer.hpp>
#include <vector>

namespace HugeCTR {
/**
 * @brief
 * This class implements the fully connected layer.
 */
class FullyConnectedLayer : public Layer {
 private:
  const bool use_mixed_precision_{false};
  const bool enable_tf32_compute_{false};
  // Optimized cublasGemmEx algorithm selection
  cublasGemmAlgo_t falgo_{CUBLAS_GEMM_DEFAULT};
  cublasGemmAlgo_t balgo_W_{CUBLAS_GEMM_DEFAULT};
  cublasGemmAlgo_t balgo_Xn_{CUBLAS_GEMM_DEFAULT};

  /*
   * stores the weight tensors of this layer.
   */
  // Tensors<float> weights_; It is inherited from Layer, and named as weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors2<float> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<float> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<float> out_tensors_;

  Tensors2<float>& get_in_tensors(bool is_train) { return in_tensors_; }

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
  /**
   * This is the constructor of the FullyConnectedLayer.
   * It will check whether the format combination of all tensors is supported or not.
   * Only two kinds of tensor formats are supported:
   * (1) weight, input, output, wgrad are all in row-major.
   * (2) weight, input, output, wgrad are all in column-major.
   * @param weight_buff: stores the weight tensor
   * @param wgrad_buff: stores the gradient values of the weight calculated in backward pass
   * @param in_tensor: stores the input tensor
   * @param out_tensor: stores the output tensor
   * @param weight_format: specifies the format of the weight tensor, either HW (row major) or WH
   * (col-major)
   */
  FullyConnectedLayer(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                      const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                      const Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource, bool use_mixed_precision,
                      bool enable_tf32_compute,
                      std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  FullyConnectedLayer(const FullyConnectedLayer& C) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&);

 private:
  /*
   * initializers for this layer.
   */
  std::unique_ptr<DataSimulator> get_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_uniform_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_xavier_norm_initializer(const int index) override;
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;
};
}  // namespace HugeCTR
