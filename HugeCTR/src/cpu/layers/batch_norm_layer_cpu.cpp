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

#include <algorithm>
#include <functional>
#include <cpu/layers/batch_norm_layer_cpu.hpp>
#include <string>
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

constexpr float eps = 1e-4; // Epsilon for CPU computation

template <typename T>
void batch_norm_fprop_cpu(const float* gamma, const float* beta, const T* in, T* out,
                          int batch_size, int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      mean += in[idx];
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float in_norm = (in[idx] - mean) / sqrt(var + eps);
      out[idx] = gamma[j] * in_norm + beta[j];
    }
  }
}

template <>
void batch_norm_fprop_cpu<__half>(const float* gamma, const float* beta, const __half* in,
                                  __half* out, int batch_size, int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      mean += __half2float(in[idx]);
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float diff = __half2float(in[idx]) - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float in_norm = (__half2float(in[idx]) - mean) / sqrt(var + eps);
      out[idx] = __float2half(gamma[j] * in_norm + beta[j]);
    }
  }
}

template <typename T>
void batch_norm_bprop_cpu(const float* gamma, const T* out, T* in, int batch_size,
                          int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      mean += in[idx];
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    float inv_std = 1.0f / sqrt(var + eps);

    float d_var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float val = (out[idx] * gamma[j]) * (in[idx] - mean);
      d_var += val;
    }
    d_var *= (-0.5f) * pow(inv_std, 3);

    float val1 = 0.0f;
    float val2 = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      val1 += (out[idx] * gamma[j]);
      val2 += (in[idx] - mean);
    }
    val1 *= (-inv_std);
    val2 *= (d_var / batch_size) * -2;
    float d_mean = (val1 + val2);

    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      in[idx] = (out[idx] * gamma[j]) * inv_std + d_var * (2.0 / batch_size) * (in[idx] - mean) +
                d_mean / batch_size;
    }
  }
}

template <>
void batch_norm_bprop_cpu<__half>(const float* gamma, const __half* out, __half* in, int batch_size,
                                  int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      mean += __half2float(in[idx]);
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    float inv_std = 1.0f / sqrt(var + eps);

    float d_var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      float val = (__half2float(out[idx]) * gamma[j]) * (__half2float(in[idx]) - mean);
      d_var += val;
    }
    d_var *= (-0.5f) * pow(inv_std, 3);

    float val1 = 0.0f;
    float val2 = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      val1 += (__half2float(out[idx]) * gamma[j]);
      val2 += __half2float(in[idx] - mean);
    }
    val1 *= (-inv_std);
    val2 *= (d_var / batch_size) * -2;
    float d_mean = (val1 + val2);

    for (int i = 0; i < batch_size; i++) {
      int idx = i * num_feature + j;
      in[idx] = __float2half((__half2float(out[idx]) * gamma[j]) * inv_std +
                             d_var * (2.0 / batch_size) * (__half2float(in[idx]) - mean) +
                             d_mean / batch_size);
    }
  }
}

} // end namespace

template <typename T>
BatchNormLayerCPU<T>::BatchNormLayerCPU(const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                                  const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                                  const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blob_buff,
                                  const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                                  const Params& params)
    : LayerCPU(),
      params_(params) {
  const auto& in_tensor_dim = in_tensor.get_dimensions();

  size_t num_feature = in_tensor_dim[1];

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);

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
}

template <typename T>
BatchNormLayerCPU<T>::~BatchNormLayerCPU() {}

template <typename T>
void BatchNormLayerCPU<T>::initialize() {}

template <typename T>
void BatchNormLayerCPU<T>::fprop(bool is_train) {
  int batch_size = in_tensors_[0].get_dimensions()[0];
  int num_feature = in_tensors_[0].get_dimensions()[1];

  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  T* in = in_tensor.get_ptr();
  T* out = out_tensor.get_ptr();

  float* gamma = gamma_.get_ptr();
  float* beta = beta_.get_ptr();

  batch_norm_fprop_cpu<T>(gamma, beta, in, out, batch_size, num_feature);

}

template <typename T>
void BatchNormLayerCPU<T>::bprop() {}

template class BatchNormLayerCPU<float>;
template class BatchNormLayerCPU<__half>;

}  // namespace HugeCTR
