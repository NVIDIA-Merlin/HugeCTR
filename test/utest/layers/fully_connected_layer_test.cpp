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

#include "HugeCTR/include/layers/fully_connected_layer.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

#include "utest/test_utils.h"
using namespace std;
using namespace HugeCTR;

static bool check_cpu_gpu(float *cpu_p, float *gpu_p, int len, float tol) {
  float *cpu_tmp = (float *)malloc(sizeof(float) * len);
  CK_CUDA_THROW_(cudaMemcpy(cpu_tmp, gpu_p, sizeof(float) * len, cudaMemcpyDeviceToHost));
  float max_diff = fabs(cpu_p[0] - cpu_tmp[0]);
  bool flag = true;
  for (int i = 0; i < len; ++i) {
    if (fabs(cpu_p[i] - cpu_tmp[i]) >= tol) flag = false;
    max_diff = max(max_diff, fabs(cpu_p[i] - cpu_tmp[i]));
  }
  free(cpu_tmp);
  return flag;
}

static void cpu_mm(float *a, float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
      for (int kk = 0; kk < k; ++kk) c[i * n + j] += a[i * k + kk] * b[kk * n + j];
    }
  }
}

static void cpu_add_bias(float *out, float *bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      out[i * n + j] += bias[j];
    }
  }
}

static void transpose(float *a, int m, int n) {
  std::unique_ptr<float[]> tmp(new float[m * n]);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) tmp[j * m + i] = a[i * n + j];
  for (int i = 0; i < m * n; ++i) a[i] = tmp[i];
}

static void fully_connected_layer_test(size_t m, size_t n, size_t k, float tol = 1e-3,
                                       bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> blobs_buff =
      GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> weight_buff = blobs_buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = blobs_buff->create_block<float>();

  Tensor2<float> in_tensor;
  blobs_buff->reserve({m, k}, &in_tensor);
  Tensor2<float> out_tensor;
  blobs_buff->reserve({m, n}, &out_tensor);

  FullyConnectedLayer<float> fully_connected_layer(weight_buff, wgrad_buff, in_tensor, out_tensor,
                                                   test::get_default_gpu(), false,
                                                   enable_tf32_compute);
  // Initialize tensors to 0 and choose cublas algorithms
  blobs_buff->allocate();
  fully_connected_layer.initialize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  Tensor2<float> weight = weight_buff->as_tensor();
  Tensor2<float> wgrad = wgrad_buff->as_tensor();

  CK_CUDA_THROW_(cudaMemset(weight.get_ptr(), 0, weight.get_size_in_bytes()));
  CK_CUDA_THROW_(cudaMemset(wgrad.get_ptr(), 0, wgrad.get_size_in_bytes()));

  // TODO: result check
  float *d_weight = weight.get_ptr();
  float *d_weight_grad = wgrad.get_ptr();
  float *d_in = in_tensor.get_ptr();
  float *d_out = out_tensor.get_ptr();

  std::unique_ptr<float[]> h_weight(new float[test::align_to_even(n * k)]);
  std::unique_ptr<float[]> h_weight_grad(new float[n * k]);
  std::unique_ptr<float[]> h_bias_grad(new float[n]);
  std::unique_ptr<float[]> h_in(new float[test::align_to_even(k * m)]);
  std::unique_ptr<float[]> h_out(new float[test::align_to_even(n * m)]);
  std::unique_ptr<float[]> h_bias(new float[test::align_to_even(n)]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  simulator.fill(h_weight.get(), test::align_to_even(k * n));
  simulator.fill(h_in.get(), test::align_to_even(m * k));
  simulator.fill(h_bias.get(), test::align_to_even(n));

  // cpu fprop
  cpu_mm(h_in.get(), h_weight.get(), h_out.get(), m, k, n);
  cpu_add_bias(h_out.get(), h_bias.get(), m, n);

  CK_CUDA_THROW_(
      cudaMemcpy(d_weight, h_weight.get(), sizeof(float) * k * n, cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(
      cudaMemcpy(d_weight + k * n, h_bias.get(), sizeof(float) * n, cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), sizeof(float) * m * k, cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fully_connected_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_out.get(), d_out, m * n, tol))
      << "fprop cross_check result fail" << endl;

  simulator.fill(h_out.get(), test::align_to_even(m * n));

  for (size_t i = 0; i < n; ++i) h_bias_grad[i] = 0.0f;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) h_bias_grad[j] += h_out[i * n + j];
  }
  // CPU bprop
  transpose(h_weight.get(), k, n);
  transpose(h_in.get(), m, k);
  cpu_mm(h_in.get(), h_out.get(), h_weight_grad.get(), k, m, n);
  cpu_mm(h_out.get(), h_weight.get(), h_in.get(), m, n, k);

  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fully_connected_layer.bprop();
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_in.get(), d_in, m * k, tol))
      << " bprop cross_check input_grad fail" << endl;
  ASSERT_EQ(true, check_cpu_gpu(h_weight_grad.get(), d_weight_grad, k * n, tol))
      << " bprop cross_check weight_grad fail" << endl;
  ASSERT_EQ(true, check_cpu_gpu(h_bias_grad.get(), d_weight_grad + k * n, n, tol))
      << " bprop cross_check bias_grad fail" << endl;
}

TEST(fully_connected_layer, fp32_1024x1024x1024) { fully_connected_layer_test(1024, 1024, 1024); }
TEST(fully_connected_layer, fp32_2048x2048x2048) { fully_connected_layer_test(2048, 2048, 2048); }
TEST(fully_connected_layer, fp32_1x1024x1024) { fully_connected_layer_test(1, 1024, 1024); }
TEST(fully_connected_layer, fp32_1024x1x1024) { fully_connected_layer_test(1024, 1, 1024); }
TEST(fully_connected_layer, fp32_1024x1024x1) { fully_connected_layer_test(1024, 1024, 1); }
TEST(fully_connected_layer, fp32_1x1x1) { fully_connected_layer_test(1, 1, 1); }
TEST(fully_connected_layer, fp32_256x512x1024) { fully_connected_layer_test(256, 512, 1024); }
TEST(fully_connected_layer, fp32_251x511x1023) { fully_connected_layer_test(251, 511, 1023); }
TEST(fully_connected_layer, tf32_256x512x1024) {
  fully_connected_layer_test(256, 512, 1024, 5e-0, true);
}
TEST(fully_connected_layer, tf32_251x511x1023) {
  fully_connected_layer_test(251, 511, 1023, 5e-0, true);
}
