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

#include <algorithm>
#include <functional>
#include <layers/batch_norm_layer.hpp>
#include <string>
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
using ToStringType = typename std::conditional<std::is_same<T, __half>::value, float, T>::type;
}

template <typename T>
BatchNormLayer<T>::BatchNormLayer(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                                  const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                                  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blob_buff,
                                  const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                                  const Params& params,
                                  const std::shared_ptr<GPUResource>& gpu_resource,
                                  std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      params_(params),
      mode_(CUDNN_BATCHNORM_PER_ACTIVATION) {
  CudaDeviceContext context(get_device_id());
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  assert(get_size_from_dims(in_tensor_dim) == get_size_from_dims(out_tensor_dim));
  assert(in_tensor_dim.size() == 2 && out_tensor_dim.size() == 2);
  assert(in_tensor_dim[0] == out_tensor_dim[0]);
  assert(in_tensor_dim[1] == out_tensor_dim[1]);

  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&in_out_desc_));

  size_t num_feature = in_tensor_dim[1];
  int batch_size = in_tensor_dim[0];

  cudnnDataType_t data_type = std::is_same<T, __half>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
  int n_stride = num_feature;
  int w_stride = 1;

  CK_CUDNN_THROW_(cudnnSetTensor4dDescriptorEx(in_out_desc_, data_type, batch_size, 1, 1,
                                               num_feature, n_stride, 1, 1, w_stride));

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);

  CK_CUDNN_THROW_(cudnnCreateTensorDescriptor(&gamma_beta_desc_));

  CK_CUDNN_THROW_(cudnnDeriveBNTensorDescriptor(gamma_beta_desc_, in_out_desc_, mode_));

  std::vector<size_t> gamma_dim = {num_feature, 1};

  // gamma & beta
  weight_buff->reserve(gamma_dim, &gamma_);
  weight_buff->reserve(gamma_dim, &beta_);
  weights_.push_back(gamma_);
  weights_.push_back(beta_);

  // gamma grad & beta grad
  wgrad_buff->reserve(gamma_dim, &gamma_grad_);
  wgrad_buff->reserve(gamma_dim, &beta_grad_);
  wgrad_.push_back(gamma_grad_);
  wgrad_.push_back(beta_grad_);

  blob_buff->reserve(in_tensor_dim, &temp_in_tensor_);

  // result running mean & var
  blob_buff->reserve(gamma_dim, &result_running_mean_);
  blob_buff->reserve(gamma_dim, &result_running_var_);

  // save running mean & var (cache)
  blob_buff->reserve(gamma_dim, &result_save_mean_);
  blob_buff->reserve(gamma_dim, &result_save_inv_var_);
}

template <typename T>
BatchNormLayer<T>::~BatchNormLayer() {
  try {
    CK_CUDNN_THROW_(cudnnDestroyTensorDescriptor(in_out_desc_));
    CK_CUDNN_THROW_(cudnnDestroyTensorDescriptor(gamma_beta_desc_));
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  }
}

template <typename T>
void BatchNormLayer<T>::initialize() {
  // host array to get running mean & var

  size_t num_feature = in_tensors_[0].get_dimensions()[1];

  std::shared_ptr<GeneralBuffer2<HostAllocator>> internal_host_buf =
      GeneralBuffer2<HostAllocator>::create();

  internal_host_buf->reserve({num_feature}, &h_result_running_mean_);
  internal_host_buf->reserve({num_feature}, &h_result_running_var_);

  internal_host_buf->allocate();
}

template <typename T>
void BatchNormLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  float one = 1.0f, zero = 0.0f;

  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();

  float* gamma = gamma_.get_ptr();
  float* beta = beta_.get_ptr();

  float* result_running_mean = result_running_mean_.get_ptr();
  float* result_running_var = result_running_var_.get_ptr();
  float* result_save_mean = result_save_mean_.get_ptr();
  float* result_save_inv_var = result_save_inv_var_.get_ptr();

  if (is_train) {
    CK_CUDNN_THROW_(cudnnBatchNormalizationForwardTraining(
        get_gpu().get_cudnn_handle(), mode_, &one, &zero, in_out_desc_, in, in_out_desc_, out,
        gamma_beta_desc_, gamma, beta, params_.factor, result_running_mean, result_running_var,
        params_.eps, result_save_mean, result_save_inv_var));
  } else {
    CK_CUDNN_THROW_(cudnnBatchNormalizationForwardInference(
        get_gpu().get_cudnn_handle(), mode_, &one, &zero, in_out_desc_, in, in_out_desc_, out,
        gamma_beta_desc_, gamma, beta, result_running_mean, result_running_var, params_.eps));
  }
}

template <typename T>
void BatchNormLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  float one = 1.0f, zero = 0.0f;

  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();

  float* gamma = gamma_.get_ptr();

  float* gamma_grad = gamma_grad_.get_ptr();
  float* beta_grad = beta_grad_.get_ptr();

  float* result_save_mean = result_save_mean_.get_ptr();
  float* result_save_inv_var = result_save_inv_var_.get_ptr();

  T* temp_in = temp_in_tensor_.get_ptr();
  size_t n_byte = temp_in_tensor_.get_size_in_bytes();
  CK_CUDA_THROW_(
      cudaMemcpyAsync(temp_in, in, n_byte, cudaMemcpyDeviceToDevice, get_gpu().get_stream()));

  CK_CUDNN_THROW_(cudnnBatchNormalizationBackward(
      get_gpu().get_cudnn_handle(), mode_, &one, &zero, &one, &zero, in_out_desc_, temp_in,
      in_out_desc_, out, in_out_desc_, in, gamma_beta_desc_, gamma, gamma_grad, beta_grad,
      params_.eps, result_save_mean, result_save_inv_var));
}

template <typename T>
std::string BatchNormLayer<T>::get_no_trained_params_in_string() {
  float* d_result_running_mean = result_running_mean_.get_ptr();
  float* d_result_running_var = result_running_var_.get_ptr();
  size_t n_byte = result_running_mean_.get_size_in_bytes();
  size_t n_elem = n_byte / sizeof(T);

  CK_CUDA_THROW_(cudaMemcpy(h_result_running_mean_.get_ptr(), d_result_running_mean, n_byte,
                            cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(cudaMemcpy(h_result_running_var_.get_ptr(), d_result_running_var, n_byte,
                            cudaMemcpyDeviceToHost));

  std::string result = "      \"type\": \"BatchNorm\",\n";
  result += "      \"mean\": [";
  for (size_t i = 0; i < n_elem; i++) {
    result += std::to_string(ToStringType<T>(h_result_running_mean_.get_ptr()[i]));
    if (i != (n_elem - 1)) result += ", ";
  }
  result += "],\n";

  result += "      \"var\": [";
  for (size_t i = 0; i < n_elem; i++) {
    result += std::to_string(ToStringType<T>(h_result_running_var_.get_ptr()[i]));
    if (i != (n_elem - 1)) result += ", ";
  }
  result += "]";

  return result;
}

template <typename T>
std::unique_ptr<DataSimulator> BatchNormLayer<T>::get_default_initializer(const int index) {
  std::unique_ptr<DataSimulator> simu;
  if (0 == index) {
    simu.reset(new ConstantDataSimulator(1.0f));
  } else if (1 == index) {
    simu.reset(new ConstantDataSimulator(0.0f));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }
  return simu;
}

template class BatchNormLayer<float>;
template class BatchNormLayer<__half>;

}  // namespace HugeCTR
