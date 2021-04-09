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

#include <math.h>

#include <cpu/layers/fully_connected_layer_cpu.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

namespace {

void cpu_mm(float *a, float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
      for (int kk = 0; kk < k; ++kk) c[i * n + j] += a[i * k + kk] * b[kk * n + j];
    }
  }
}

void cpu_add_bias(float *out, float *bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      out[i * n + j] += bias[j];
    }
  }
}

void transpose(float *a, int m, int n) {
  std::unique_ptr<float[]> tmp(new float[m * n]);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) tmp[j * m + i] = a[i * n + j];
  for (int i = 0; i < m * n; ++i) a[i] = tmp[i];
}

} // end namespace

FullyConnectedLayerCPU<float>::FullyConnectedLayerCPU(
    const std::shared_ptr<BufferBlock2<float>>& weight_buff,
    const std::shared_ptr<BufferBlock2<float>>& wgrad_buff, const Tensor2<float>& in_tensor,
    const Tensor2<float>& out_tensor, bool use_mixed_precision)
    : LayerCPU(),
      use_mixed_precision_(use_mixed_precision) {
  try {
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.get_dimensions();
    const auto& out_tensor_dim = out_tensor.get_dimensions();
    // 1. two dim?
    if (in_tensor_dim.size() != 2 || out_tensor_dim.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
    }
    // 2. dim match?
    size_t m = in_tensor_dim[0];
    size_t n = out_tensor_dim[1];
    size_t k = in_tensor_dim[1];
    size_t m_ck = out_tensor_dim[0];
    if (m != m_ck) {
      CK_THROW_(Error_t::WrongInput, "size of input / output tensor doesn't match");
    }

    std::vector<size_t> weight_dim = {k, n};
    std::vector<size_t> bias_dim = {1, n};

    {
      Tensor2<float> tensor;
      weight_buff->reserve(weight_dim, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      weight_buff->reserve(bias_dim, &tensor);
      weights_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      wgrad_buff->reserve(weight_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    {
      Tensor2<float> tensor;
      wgrad_buff->reserve(bias_dim, &tensor);
      wgrad_.push_back(tensor);
    }
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);
    // Where should we create this cuBLAS handle?
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void FullyConnectedLayerCPU<float>::fprop(bool is_train) {
  Tensor2<float>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<float>& out_tensor = out_tensors_[0];

  float* weight = weights_[0].get_ptr();
  float* bias = weights_[1].get_ptr();
  float* in = in_tensor.get_ptr();
  float* out = out_tensor.get_ptr();

  const auto& in_tensor_dim = in_tensor.get_dimensions();
  const auto& out_tensor_dim = out_tensor.get_dimensions();

  int m, n, k;

  m = in_tensor_dim[0];
  n = out_tensor_dim[1];
  k = in_tensor_dim[1];

  cpu_mm(in, weight, out, m, k, n);
  cpu_add_bias(out, bias, m, n);
}

void FullyConnectedLayerCPU<float>::bprop() {}

template class FullyConnectedLayerCPU<float>;

}  // namespace HugeCTR
