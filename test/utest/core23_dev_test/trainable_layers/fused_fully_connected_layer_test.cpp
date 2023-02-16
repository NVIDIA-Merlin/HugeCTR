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

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core23/tensor_container.hpp>
#include <cstdlib>
#include <layers/fused_fully_connected_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

static void cpu_mm(__half *c, const __half *a, bool transpose_a, const __half *b, bool transpose_b,
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

static void cpu_add_bias_and_re(__half *top, __half *middle, const __half *bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __half t = top[i * n + j] + bias[j];
      middle[i * n + j] = t;
      top[i * n + j] = t < 0 ? __float2half(0.0f) : t;
    }
  }
}

static void cpu_reverse_add_bias_and_re(__half *bias_grad, __half *middle, const __half *top, int m,
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

static float compare_array(const __half *arr1, const __half *arr2, size_t n, float threshold) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    if (fabs(arr1[i] - arr2[i]) > threshold) {
      m++;
    }
  }
  return 1.0f * m / n;
}

static void fully_connected_layer_test(int64_t m, int64_t n, int64_t k) {
  HCTR_LOG(INFO, WORLD, "Testing m=%zu, n=%zu, k=%zu\n", m, n, k);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  core23::Tensor bottom_tensor = core23::Tensor(core23::TensorParams()
                                                    .data_type(core23::ToScalarType<__half>::value)
                                                    .shape({m, k})
                                                    .buffer_params(blobs_buffer_params));

  core23::Tensor top_tensor = core23::Tensor(core23::TensorParams()
                                                 .data_type(core23::ToScalarType<__half>::value)
                                                 .shape({m, n})
                                                 .buffer_params(blobs_buffer_params));

  Core23TempFusedFullyConnectedLayer fully_connected_layer(bottom_tensor, top_tensor,
                                                           test::get_default_gpu());

  // Initialize tensors to 0 and choose cublas algorithms
  fully_connected_layer.initialize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)

  auto weights = fully_connected_layer.get_weights();
  auto weights_grad = fully_connected_layer.get_wgrads();

  core23::TensorContainer<__half, 1, 1> weights_container(std::move(weights),
                                                          {static_cast<int64_t>(weights.size())});
  core23::TensorContainer<__half, 1, 1> weights_grad_container(
      std::move(weights_grad), {static_cast<int64_t>(weights_grad.size())});
  auto weights_view = weights_container.flatten();
  auto weights_grad_view = weights_grad_container.flatten();

  HCTR_LIB_THROW(cudaMemset(weights_view.data(), 0.0, sizeof(__half) * weights_view.size(0)));
  HCTR_LIB_THROW(
      cudaMemset(weights_grad_view.data(), 0.0, sizeof(__half) * weights_grad_view.size(0)));

  // TODO: result check
  __half *d_kernel = weights_container[0].template data<__half>();
  __half *d_bias = weights_container[1].template data<__half>();
  __half *d_kernel_grad = weights_grad_container[0].template data<__half>();
  __half *d_bias_grad = weights_grad_container[1].template data<__half>();
  __half *d_bottom = bottom_tensor.template data<__half>();
  __half *d_top = top_tensor.template data<__half>();

  std::unique_ptr<__half[]> h_kernel(new __half[k * n]);
  std::unique_ptr<__half[]> h_kernel_grad(new __half[k * n]);
  std::unique_ptr<__half[]> h_bias_grad(new __half[n]);
  std::unique_ptr<__half[]> h_bottom(new __half[m * k]);
  std::unique_ptr<__half[]> h_middle(new __half[m * n]);
  std::unique_ptr<__half[]> h_top(new __half[m * n]);
  std::unique_ptr<__half[]> h_bias(new __half[n]);

  std::unique_ptr<__half[]> d2h_top(new __half[m * n]);
  std::unique_ptr<__half[]> d2h_bottom(new __half[m * k]);
  std::unique_ptr<__half[]> d2h_kernel_grad(new __half[k * n]);
  std::unique_ptr<__half[]> d2h_bias_grad(new __half[n]);

  simulator.fill(h_bottom.get(), m * k);
  simulator.fill(h_kernel.get(), k * n);
  simulator.fill(h_bias.get(), n);

  HCTR_LIB_THROW(
      cudaMemcpy(d_kernel, h_kernel.get(), sizeof(__half) * k * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_bias, h_bias.get(), sizeof(__half) * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_bottom, h_bottom.get(), sizeof(__half) * m * k, cudaMemcpyHostToDevice));

  // cpu fprop
  cpu_mm(h_top.get(), h_bottom.get(), false, h_kernel.get(), false, m, k, n);
  cpu_add_bias_and_re(h_top.get(), h_middle.get(), h_bias.get(), m, n);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(cudaMemcpy(d2h_top.get(), d_top, sizeof(__half) * m * n, cudaMemcpyDeviceToHost));

  ASSERT_LT(compare_array(h_top.get(), d2h_top.get(), m * n, 1e-5), 0.01f)
      << "fprop cross_check result fail" << std::endl;

  simulator.fill(h_top.get(), m * n);

  HCTR_LIB_THROW(cudaMemcpy(d_top, h_top.get(), sizeof(__half) * m * n, cudaMemcpyHostToDevice));

  cpu_reverse_add_bias_and_re(h_bias_grad.get(), h_middle.get(), h_top.get(), m, n);

  cpu_mm(h_kernel_grad.get(), h_bottom.get(), true, h_middle.get(), false, k, m, n);
  cpu_mm(h_bottom.get(), h_middle.get(), false, h_kernel.get(), true, m, n, k);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  HCTR_LIB_THROW(
      cudaMemcpy(d2h_bottom.get(), d_bottom, sizeof(__half) * m * k, cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(d2h_kernel_grad.get(), d_kernel_grad, sizeof(__half) * k * n,
                            cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_bias_grad.get(), d_bias_grad, sizeof(__half) * n, cudaMemcpyDeviceToHost));

  ASSERT_LT(compare_array(h_bottom.get(), d2h_bottom.get(), m * k, 1e-1), 0.05f)
      << " bprop cross_check input_grad fail" << std::endl;
  ASSERT_LT(compare_array(h_kernel_grad.get(), d2h_kernel_grad.get(), k * n, 1e-1), 0.15f)
      << " bprop cross_check weight_grad fail" << std::endl;
  ASSERT_LT(compare_array(h_bias_grad.get(), d2h_bias_grad.get(), n, 1e-1), 0.15f)
      << " bprop cross_check bias_grad fail" << std::endl;
}

TEST(fused_fully_connected_layer, fp16_32x64x32) { fully_connected_layer_test(32, 64, 32); }
TEST(fused_fully_connected_layer, fp16_2048x512x16) { fully_connected_layer_test(2048, 512, 16); }
TEST(fused_fully_connected_layer, fp16_2048x1024x480) {
  fully_connected_layer_test(2048, 1024, 480);
}
TEST(fused_fully_connected_layer, fp16_2048x512x1024) {
  fully_connected_layer_test(2048, 512, 1024);
}
TEST(fused_fully_connected_layer, fp16_2048x1024x1024) {
  fully_connected_layer_test(2048, 1024, 1024);
}
