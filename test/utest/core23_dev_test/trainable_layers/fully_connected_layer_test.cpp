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

#include <cmath>
#include <core23/tensor_container.hpp>
#include <cstdlib>
#include <layers/fully_connected_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

static bool check_cpu_gpu(float *cpu_p, float *gpu_p, int len, float tol) {
  float *cpu_tmp = (float *)malloc(sizeof(float) * len);
  HCTR_LIB_THROW(cudaMemcpy(cpu_tmp, gpu_p, sizeof(float) * len, cudaMemcpyDeviceToHost));
  float max_diff = fabs(cpu_p[0] - cpu_tmp[0]);
  bool flag = true;
  for (int i = 0; i < len; ++i) {
    if (fabs(cpu_p[i] - cpu_tmp[i]) >= tol) {
      flag = false;
    }
    max_diff = std::max(max_diff, fabs(cpu_p[i] - cpu_tmp[i]));
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

static void fully_connected_layer_test(int64_t m, int64_t n, int64_t k, float tol = 1e-3,
                                       bool enable_tf32_compute = false) {
  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  core23::Tensor in_tensor = core23::Tensor(core23::TensorParams()
                                                .data_type(core23::ToScalarType<float>::value)
                                                .shape({m, k})
                                                .buffer_params(blobs_buffer_params));

  core23::Tensor out_tensor = core23::Tensor(core23::TensorParams()
                                                 .data_type(core23::ToScalarType<float>::value)
                                                 .shape({m, n})
                                                 .buffer_params(blobs_buffer_params));

  Core23TempFullyConnectedLayer<float> fully_connected_layer(
      in_tensor, out_tensor, test::get_default_gpu(), false, enable_tf32_compute);
  fully_connected_layer.initialize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  auto weights = fully_connected_layer.get_weights();
  auto weights_grad = fully_connected_layer.get_wgrads();

  core23::TensorContainer<float, 1, 1> weights_container(std::move(weights),
                                                         {static_cast<int64_t>(weights.size())});
  core23::TensorContainer<float, 1, 1> weights_grad_container(
      std::move(weights_grad), {static_cast<int64_t>(weights_grad.size())});
  auto weights_view = weights_container.flatten();
  auto weights_grad_view = weights_grad_container.flatten();

  HCTR_LIB_THROW(cudaMemset(weights_view.data(), 0.0, sizeof(float) * weights_view.size(0)));
  HCTR_LIB_THROW(
      cudaMemset(weights_grad_view.data(), 0.0, sizeof(float) * weights_grad_view.size(0)));

  // TODO: result check
  float *d_weight = weights_container[0].data<float>();
  float *d_bias = weights_container[1].data<float>();
  float *d_weight_grad = weights_grad_container[0].data<float>();
  float *d_bias_grad = weights_grad_container[1].data<float>();
  float *d_in = in_tensor.data<float>();
  float *d_out = out_tensor.data<float>();

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

  HCTR_LIB_THROW(
      cudaMemcpy(d_weight, h_weight.get(), sizeof(float) * k * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_bias, h_bias.get(), sizeof(float) * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), sizeof(float) * m * k, cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_out.get(), d_out, m * n, tol))
      << "fprop cross_check result fail" << std::endl;

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

  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_in.get(), d_in, m * k, tol))
      << " bprop cross_check input_grad fail" << std::endl;
  ASSERT_EQ(true, check_cpu_gpu(h_weight_grad.get(), d_weight_grad, k * n, tol))
      << " bprop cross_check weight_grad fail" << std::endl;
  ASSERT_EQ(true, check_cpu_gpu(h_bias_grad.get(), d_bias_grad, n, tol))
      << " bprop cross_check bias_grad fail" << std::endl;
}

static void fully_connected_layer_test_3d(int64_t batch_size, int64_t seq_len, int64_t n, int64_t k,
                                          float tol = 1e-3, bool enable_tf32_compute = false) {
  core23::BufferParams blobs_buffer_params = {};
  // blobs_buffer_params.channel = GetBlobsBufferChannel();
  blobs_buffer_params.channel = core23::GetRandomBufferChannel();

  core23::Tensor in_tensor = core23::Tensor(core23::TensorParams()
                                                .data_type(core23::ToScalarType<float>::value)
                                                .shape({batch_size, seq_len, k})
                                                .buffer_params(blobs_buffer_params));

  core23::Tensor out_tensor = core23::Tensor(core23::TensorParams()
                                                 .data_type(core23::ToScalarType<float>::value)
                                                 .shape({batch_size, seq_len, n})
                                                 .buffer_params(blobs_buffer_params));

  Core23TempFullyConnectedLayer<float> fully_connected_layer(
      in_tensor, out_tensor, test::get_default_gpu(), false, enable_tf32_compute);
  // Initialize tensors to 0 and choose cublas algorithms
  fully_connected_layer.initialize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  auto weights = fully_connected_layer.get_weights();
  auto weights_grad = fully_connected_layer.get_wgrads();

  core23::TensorContainer<float, 1, 1> weights_container(std::move(weights),
                                                         {static_cast<int64_t>(weights.size())});
  core23::TensorContainer<float, 1, 1> weights_grad_container(
      std::move(weights_grad), {static_cast<int64_t>(weights_grad.size())});
  auto weights_view = weights_container.flatten();
  auto weights_grad_view = weights_grad_container.flatten();

  HCTR_LIB_THROW(cudaMemset(weights_view.data(), 0.0, sizeof(float) * weights_view.size(0)));
  HCTR_LIB_THROW(
      cudaMemset(weights_grad_view.data(), 0.0, sizeof(float) * weights_grad_view.size(0)));

  // TODO: result check
  float *d_weight = weights_container[0].data<float>();
  float *d_bias = weights_container[1].data<float>();
  float *d_weight_grad = weights_grad_container[0].data<float>();
  float *d_bias_grad = weights_grad_container[1].data<float>();
  float *d_in = in_tensor.data<float>();
  float *d_out = out_tensor.data<float>();

  std::unique_ptr<float[]> h_weight(new float[test::align_to_even(n * k)]);
  std::unique_ptr<float[]> h_weight_grad(new float[n * k]);
  std::unique_ptr<float[]> h_bias_grad(new float[n]);
  std::unique_ptr<float[]> h_in(new float[test::align_to_even(batch_size * seq_len * k)]);
  std::unique_ptr<float[]> h_out(new float[test::align_to_even(batch_size * seq_len * n)]);
  std::unique_ptr<float[]> h_bias(new float[test::align_to_even(n)]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  simulator.fill(h_weight.get(), test::align_to_even(k * n));
  simulator.fill(h_in.get(), test::align_to_even(batch_size * seq_len * k));
  simulator.fill(h_bias.get(), test::align_to_even(n));

  // cpu fprop
  cpu_mm(h_in.get(), h_weight.get(), h_out.get(), batch_size * seq_len, k, n);
  cpu_add_bias(h_out.get(), h_bias.get(), batch_size * seq_len, n);

  HCTR_LIB_THROW(
      cudaMemcpy(d_weight, h_weight.get(), sizeof(float) * k * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_bias, h_bias.get(), sizeof(float) * n, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(d_in, h_in.get(), sizeof(float) * batch_size * seq_len * k,
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_out.get(), d_out, batch_size * seq_len * n, tol))
      << "fprop cross_check result fail" << std::endl;

  simulator.fill(h_out.get(), test::align_to_even(batch_size * seq_len * n));

  for (size_t i = 0; i < n; ++i) h_bias_grad[i] = 0.0f;
  for (size_t i = 0; i < batch_size * seq_len; ++i) {
    for (size_t j = 0; j < n; ++j) h_bias_grad[j] += h_out[i * n + j];
  }
  // CPU bprop
  transpose(h_weight.get(), k, n);
  transpose(h_in.get(), batch_size * seq_len, k);
  cpu_mm(h_in.get(), h_out.get(), h_weight_grad.get(), k, batch_size * seq_len, n);
  cpu_mm(h_out.get(), h_weight.get(), h_in.get(), batch_size * seq_len, n, k);

  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), sizeof(float) * batch_size * seq_len * n,
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fully_connected_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  ASSERT_EQ(true, check_cpu_gpu(h_in.get(), d_in, batch_size * seq_len * k, tol))
      << " bprop cross_check input_grad fail" << std::endl;
  ASSERT_EQ(true, check_cpu_gpu(h_weight_grad.get(), d_weight_grad, k * n, tol))
      << " bprop cross_check weight_grad fail" << std::endl;
  ASSERT_EQ(true, check_cpu_gpu(h_bias_grad.get(), d_bias_grad, n, tol))
      << " bprop cross_check bias_grad fail" << std::endl;
}

TEST(fully_connected_layer, fp32_1024x1024x1024) { fully_connected_layer_test(1024, 1024, 1024); }
TEST(fully_connected_layer, fp32_2048x2048x2048) { fully_connected_layer_test(2048, 2048, 2048); }
TEST(fully_connected_layer, fp32_1x1024x1024) { fully_connected_layer_test(1, 1024, 1024); }
TEST(fully_connected_layer, fp32_1024x1x1024) { fully_connected_layer_test(1024, 1, 1024); }
TEST(fully_connected_layer, fp32_1024x1024x1) { fully_connected_layer_test(1024, 1024, 1); }
TEST(fully_connected_layer, fp32_1x1x1) { fully_connected_layer_test(1, 1, 1); }
TEST(fully_connected_layer, fp32_256x512x1024) { fully_connected_layer_test(256, 512, 1024); }
TEST(fully_connected_layer, fp32_251x511x1023) { fully_connected_layer_test(251, 511, 1023); }
TEST(fully_connected_layer, fp32_512x4x512x256) { fully_connected_layer_test_3d(512, 4, 512, 256); }
TEST(fully_connected_layer, fp32_512x10x512x512) {
  fully_connected_layer_test_3d(512, 10, 512, 512);
}
TEST(fully_connected_layer, tf32_256x512x1024) {
  fully_connected_layer_test(256, 512, 1024, 5e-0, true);
}
TEST(fully_connected_layer, tf32_251x511x1023) {
  fully_connected_layer_test(251, 511, 1023, 5e-0, true);
}
