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

#include "HugeCTR/include/layers/batch_norm_layer.hpp"

#include "HugeCTR/include/utils.hpp"

#include <algorithm>
#include <functional>
#include <string>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

BatchNormLayer::BatchNormLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                               const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                               const std::shared_ptr<Tensor<float>>& in_tensor,
                               const std::shared_ptr<Tensor<float>>& out_tensor,
                               const Params& params, cudnnHandle_t const& cudnn_handle,
                               int device_id,
                               std::vector<Initializer_t> initializer_types)
    : Layer(device_id, initializer_types),
      params_(params),
      mode_(CUDNN_BATCHNORM_PER_ACTIVATION),
      cudnn_handle_(cudnn_handle) {
  CudaDeviceContext context(get_device_id());
  const auto& in_tensor_dim = in_tensor->get_dims();
  const auto& out_tensor_dim = out_tensor->get_dims();
  TensorFormat_t in_format = in_tensor->get_format();
  TensorFormat_t out_format = out_tensor->get_format();

  assert(get_size_from_dims(in_tensor_dim) == get_size_from_dims(out_tensor_dim));
  assert(in_tensor_dim.size() == 2 && out_tensor_dim.size() == 2);
  assert(in_format == out_format);
  assert(in_format == TensorFormat_t::WH || in_format == TensorFormat_t::HW);
  assert(in_tensor_dim[0] == out_tensor_dim[0]);
  assert(in_tensor_dim[1] == out_tensor_dim[1]);

  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&in_out_desc_));

  bool is_column_major = (in_format == TensorFormat_t::WH);

  size_t num_feature = is_column_major ? in_tensor_dim[0] : in_tensor_dim[1];
  int batch_size = is_column_major ? in_tensor_dim[1] : in_tensor_dim[0];

  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  int n_stride = is_column_major ? 1 : num_feature;
  int w_stride = is_column_major ? batch_size : 1;

  CK_CUDNN_THROW_(cudnnSetTensor4dDescriptorEx(in_out_desc_, data_type, batch_size, 1, 1,
                                               num_feature, n_stride, 1, 1, w_stride));

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);

  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&gamma_beta_desc_));

  CK_CUDNN_THROW_(cudnnDeriveBNTensorDescriptor(gamma_beta_desc_, in_out_desc_, mode_));

  auto gamma_format = TensorFormat_t::WH;
  std::vector<size_t> gamma_dim = {num_feature, 1};

  // gamma & beta
  gamma_.reset(new Tensor<float>(gamma_dim, weight_buff, gamma_format));
  beta_.reset(new Tensor<float>(gamma_dim, weight_buff, gamma_format));
  weights_.push_back(gamma_);
  weights_.push_back(beta_);

  // gamma grad & beta grad
  gamma_grad_.reset(new Tensor<float>(gamma_dim, wgrad_buff, gamma_format));
  beta_grad_.reset(new Tensor<float>(gamma_dim, wgrad_buff, gamma_format));
  wgrad_.emplace_back(gamma_grad_);
  wgrad_.emplace_back(beta_grad_);

  std::shared_ptr<GeneralBuffer<float>> internal_buf(new GeneralBuffer<float>());

  temp_in_tensor_.reset(new Tensor<float>(in_tensor_dim, internal_buf, in_format));

  // result running mean & var
  result_running_mean_.reset(new Tensor<float>(gamma_dim, internal_buf, gamma_format));
  result_running_var_.reset(new Tensor<float>(gamma_dim, internal_buf, gamma_format));

  // save running mean & var (cache)
  result_save_mean_.reset(new Tensor<float>(gamma_dim, internal_buf, gamma_format));
  result_save_inv_var_.reset(new Tensor<float>(gamma_dim, internal_buf, gamma_format));

  internal_buf->init(get_device_id());

  // host array to get running mean & var
  h_result_running_mean_.reset(new float[num_feature]);
  h_result_running_var_.reset(new float[num_feature]);
}

BatchNormLayer::~BatchNormLayer() {
  try {
    CK_CUDNN_THROW_(cudnnDestroyTensorDescriptor(in_out_desc_));
    CK_CUDNN_THROW_(cudnnDestroyTensorDescriptor(gamma_beta_desc_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

void BatchNormLayer::inference(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, stream));
  float one = 1.0f, zero = 0.0f;

  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();

  float* gamma = gamma_->get_ptr();
  float* beta = beta_->get_ptr();

  float* result_running_mean = result_running_mean_->get_ptr();
  float* result_running_var = result_running_var_->get_ptr();
  float* result_save_mean = result_save_mean_->get_ptr();
  float* result_save_inv_var = result_save_inv_var_->get_ptr();

  CK_CUDNN_THROW_(cudnnBatchNormalizationForwardInference(
      cudnn_handle_, mode_, &one, &zero, in_out_desc_, in, in_out_desc_, out, gamma_beta_desc_,
      gamma, beta, result_running_mean, result_running_var, params_.eps));
}

void BatchNormLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, stream));

  float one = 1.0f, zero = 0.0f;

  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();

  float* gamma = gamma_->get_ptr();
  float* beta = beta_->get_ptr();

  float* result_running_mean = result_running_mean_->get_ptr();
  float* result_running_var = result_running_var_->get_ptr();
  float* result_save_mean = result_save_mean_->get_ptr();
  float* result_save_inv_var = result_save_inv_var_->get_ptr();

  CK_CUDNN_THROW_(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, mode_, &one, &zero, in_out_desc_, in, in_out_desc_, out, gamma_beta_desc_,
      gamma, beta, params_.factor, result_running_mean, result_running_var, params_.eps,
      result_save_mean, result_save_inv_var));
}

void BatchNormLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  CK_CUDNN_THROW_(cudnnSetStream(cudnn_handle_, stream));

  float one = 1.0f, zero = 0.0f;

  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];
  float* in = in_tensor->get_ptr();
  float* out = out_tensor->get_ptr();

  float* gamma = gamma_->get_ptr();

  float* gamma_grad = gamma_grad_->get_ptr();
  float* beta_grad = beta_grad_->get_ptr();

  float* result_save_mean = result_save_mean_->get_ptr();
  float* result_save_inv_var = result_save_inv_var_->get_ptr();

  float* temp_in = temp_in_tensor_->get_ptr();
  size_t n_byte = temp_in_tensor_->get_size();
  CK_CUDA_THROW_(cudaMemcpy(temp_in, in, n_byte, cudaMemcpyDeviceToDevice));

  CK_CUDNN_THROW_(cudnnBatchNormalizationBackward(
      cudnn_handle_, mode_, &one, &zero, &one, &zero, in_out_desc_, temp_in, in_out_desc_, out,
      in_out_desc_, in, gamma_beta_desc_, gamma, gamma_grad, beta_grad, params_.eps,
      result_save_mean, result_save_inv_var));
}

std::string BatchNormLayer::get_no_trained_params_in_string() {
  float* d_result_running_mean = result_running_mean_->get_ptr();
  float* d_result_running_var = result_running_var_->get_ptr();
  size_t n_byte = result_running_mean_->get_size();
  size_t n_elem = n_byte / sizeof(float);

  CK_CUDA_THROW_(cudaMemcpy(h_result_running_mean_.get(), d_result_running_mean, n_byte,
                            cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(cudaMemcpy(h_result_running_var_.get(), d_result_running_var, n_byte,
                            cudaMemcpyDeviceToHost));

  std::string result = "      \"type\": \"BatchNorm\",\n";
  result += "      \"mean\": [";
  for (size_t i = 0; i < n_elem; i++) {
    result += std::to_string(h_result_running_mean_[i]);
    if (i != (n_elem - 1)) result += ", ";
  }
  result += "],\n";

  result += "      \"var\": [";
  for (size_t i = 0; i < n_elem; i++) {
    result += std::to_string(h_result_running_var_[i]);
    if (i != (n_elem - 1)) result += ", ";
  }
  result += "]";

  return result;
}

std::unique_ptr<DataSimulator<float>> BatchNormLayer::get_default_initializer(const int index) {
  std::unique_ptr<DataSimulator<float>> simu(nullptr);
  if (0 == index) {
    auto ones_init = [] {return static_cast<float>(1); };
    simu.reset(new SingleDataSimulator<float>(ones_init));
  } else if (1 == index) {
    auto zeros_init = [] {return static_cast<float>(0); };
    simu.reset(new SingleDataSimulator<float>(zeros_init));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

}  // namespace HugeCTR
