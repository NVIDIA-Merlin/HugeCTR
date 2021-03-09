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

#include <cpu/layers/fused_fully_connected_layer_cpu.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

void cpu_mm(__half *c, const __half *a, bool transpose_a, const __half *b, bool transpose_b,
                   int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        int ai = transpose_a ? kk * m + i : i * k + kk;
        int bi = transpose_b ? j * k + kk : kk * n + j;
        sum += a[ai] * b[bi];
      }
      c[i * n + j] = sum;
    }
  }
}

void cpu_add_bias_and_re(__half *top, __half *middle, const __half *bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __half t = top[i * n + j] + bias[j];
      middle[i * n + j] = t;
      top[i * n + j] = t < 0 ? __float2half(0.0f) : t;
    }
  }
}

void cpu_reverse_add_bias_and_re(__half *bias_grad, __half *middle, const __half *top, int m,
                                        int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      if (middle[i * n + j] < 0) {
        middle[i * n + j] = 0.0f;
      } else {
        middle[i * n + j] = top[i * n + j];
      }
    }

  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < m; ++j) sum += middle[j * n + i];
    bias_grad[i] = sum;
  }
}

}  // namespace

FusedFullyConnectedLayerCPU::FusedFullyConnectedLayerCPU(
    const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
    const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
    const Tensor2<__half>& bottom_tensor, const Tensor2<__half>& top_tensor)
    : LayerCPU() {
  const auto& bottom_tensor_dim = bottom_tensor.get_dimensions();
  const auto& top_tensor_dim = top_tensor.get_dimensions();

  if (bottom_tensor_dim.size() != 2 || top_tensor_dim.size() != 2) {
    CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  if (m % 32 != 0 || n % 64 != 0) {
    CK_THROW_(Error_t::WrongInput,
              "The first dimension of bottom tensor must be a multiple of 32, the second dimension "
              "of top tensor must be a multiple of 64.");
  }

  std::vector<size_t> kernel_dim = {k, n};
  std::vector<size_t> bias_dim = {1, n};

  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(kernel_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(bias_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(kernel_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(bias_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(kernel_dim, &tensor);
    weights_grad_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(bias_dim, &tensor);
    weights_grad_.push_back(tensor);
  }

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
  blobs_buff->reserve(top_tensor_.get_dimensions(), &middle_tensor_);
  blobs_buff->reserve(bias_dim, &bias_grad_tensor_);
}

void FusedFullyConnectedLayerCPU::fprop(bool is_train) {

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* bias = weights_half_[1].get_ptr();
  const __half* bottom = get_bottom_tensor(is_train).get_ptr();
  __half* middle = middle_tensor_.get_ptr();
  __half* top = top_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor(is_train).get_dimensions();
  const auto& top_tensor_dim = top_tensor_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  cpu_mm(top, bottom, false, kernel, false, m, k, n);
  cpu_add_bias_and_re(top, middle, bias, m, n);
}

void FusedFullyConnectedLayerCPU::bprop() {}

}  // namespace HugeCTR
