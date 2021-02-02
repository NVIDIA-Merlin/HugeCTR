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

#include <cudnn.h>

#include <general_buffer2.hpp>
#include <layer.hpp>
#include <memory>

namespace HugeCTR {

/**
 * BatchNorm layer based on cuDNN
 */
template <typename T>
class BatchNormLayer : public Layer {
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
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

 public:
  /**
   * BatchNorm parameters
   */
  struct Params {
    double factor; /**<  moving average computation factor*/
    double eps;    /**< small value to avoid divide-by-zero error*/
  };

  /**
   * Ctor of BatchNormLayer.
   * @param weight_buff weight buffer for internal gamma/beta tensors
   * @param wgrad_buff gradient buffer for internal gamma/beta tensors
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param params BatchNorm parameters
   * @param cudnn_handle cuDNN handle created externally
   * @param device_id the id of GPU where this layer belongs
   */
  BatchNormLayer(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                 const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                 const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blob_buff,
                 const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, const Params& params,
                 const std::shared_ptr<GPUResource>& gpu_resource,
                 std::vector<Initializer_t> initializer_types = std::vector<Initializer_t>());
  ~BatchNormLayer() override;

  void initialize() override;

  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;

  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop() override;

  /**
   * A method to get mean and variance which are needed for inference as string.
   * Session is in charge of calling this method and store the contensts to file.
   * See Session::download_params_to_file() for more detailed information.
   */
  std::string get_no_trained_params_in_string() override;

 private:
  /**
   * A method of defining how gamma and beta are initialized.
   * Gamma is initialized to 1s while Beta is 0ed.
   * Override this function to change the initialization behavior.
   */
  std::unique_ptr<DataSimulator> get_default_initializer(const int index) override;

  const Params params_;
  const cudnnBatchNormMode_t mode_;
  cudnnTensorDescriptor_t in_out_desc_;
  cudnnTensorDescriptor_t gamma_beta_desc_;

  // these four pointers are just for convenience
  // they are deleted by Layer d'tor through the other pointer aliases: weight_ and wgrad_
  Tensor2<float> gamma_;
  Tensor2<float> beta_;
  Tensor2<float> gamma_grad_;
  Tensor2<float> beta_grad_;

  // these tensors are internal only managed by smart ptrs
  Tensor2<T> temp_in_tensor_;
  Tensor2<float> result_running_mean_;
  Tensor2<float> result_running_var_;
  Tensor2<float> result_save_mean_;
  Tensor2<float> result_save_inv_var_;

  // host array to do device-to-host copy for mean and var
  Tensor2<float> h_result_running_mean_;
  Tensor2<float> h_result_running_var_;
};

}  // namespace HugeCTR
