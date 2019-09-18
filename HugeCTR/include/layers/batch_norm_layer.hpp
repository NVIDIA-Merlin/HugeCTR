/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/tensor.hpp"

#include <cudnn.h>

#include <memory>

namespace HugeCTR {

/**
 * BatchNorm layer based on cuDNN
 */
class BatchNormLayer : public Layer {
 public:
  /**
   * BatchNorm parameters
   */
  struct Params {
    bool is_training; /**< if it is trained mode or not*/
    float factor;     /**<  moving average computation factor*/
    float eps;        /**< small value to avoid divide-by-zero error*/
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
  BatchNormLayer(GeneralBuffer<float>& weight_buff, GeneralBuffer<float>& wgrad_buff,
                 Tensor<float>& in_tensor, Tensor<float>& out_tensor, const Params& params,
                 cudnnHandle_t const& cudnn_handle, int device_id);
  ~BatchNormLayer() override;

  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * A method of implementing the forward pass of BatchNorm
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;
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
  std::vector<float> get_initializer() override;

  const Params params_;
  const cudnnBatchNormMode_t mode_;
  cudnnHandle_t const& cudnn_handle_;
  cudnnTensorDescriptor_t in_out_desc_;
  cudnnTensorDescriptor_t gamma_beta_desc_;

  // these four pointers are just for convenience
  // they are deleted by Layer d'tor through the other pointer aliases: weight_ and wgrad_
  Tensor<float>* gamma_;
  Tensor<float>* beta_;
  Tensor<float>* gamma_grad_;
  Tensor<float>* beta_grad_;

  GeneralBuffer<float> internal_buf_;

  // these tensors are internal only managed by smart ptrs
  std::unique_ptr<Tensor<float>> temp_in_tensor_;
  std::unique_ptr<Tensor<float>> result_running_mean_;
  std::unique_ptr<Tensor<float>> result_running_var_;
  std::unique_ptr<Tensor<float>> result_save_mean_;
  std::unique_ptr<Tensor<float>> result_save_inv_var_;

  // host array to do device-to-host copy for mean and var
  float* h_result_running_mean_;
  float* h_result_running_var_;
};

}  // namespace HugeCTR
