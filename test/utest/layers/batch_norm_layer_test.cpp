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

#include "HugeCTR/include/layers/batch_norm_layer.hpp"

#include "HugeCTR/include/general_buffer2.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-4;

void batch_norm_fprop_cpu(const float* gamma, const float* beta, const float* in, float* out,
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

void batch_norm_bprop_cpu(const float* gamma, const float* out, float* in, int batch_size,
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

void batch_norm_test(size_t batch_size, size_t num_feature) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> wbuff = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgbuff = buff->create_block<float>();

  vector<size_t> dims = {batch_size, num_feature};

  Tensor2<float> in_tensor;
  buff->reserve(dims, &in_tensor);
  Tensor2<float> out_tensor;
  buff->reserve(dims, &out_tensor);

  BatchNormLayer::Params params = {1.0, eps};
  BatchNormLayer batch_norm_layer(wbuff, wgbuff, buff, in_tensor, out_tensor, params,
                                  test::get_default_gpu());

  buff->allocate();
  batch_norm_layer.initialize();

  const size_t len = batch_size * num_feature;

  float* d_in = in_tensor.get_ptr();
  float* d_out = out_tensor.get_ptr();

  std::unique_ptr<float[]> h_gamma(new float[num_feature]);
  std::unique_ptr<float[]> h_beta(new float[num_feature]);
  std::unique_ptr<float[]> h_in(new float[len]);
  std::unique_ptr<float[]> h_out(new float[len]);
  std::unique_ptr<float[]> h_expected(new float[len]);

  test::GaussianDataSimulator simulator(0.0, 1.0);

  // standard normall distribution is assumed
  for (size_t j = 0; j < num_feature; j++) {
    h_gamma[j] = 1.0;
    h_beta[j] = 0.0;
  }

  Tensor2<float> weight_tensor = wbuff->as_tensor();

  float* d_gamma = weight_tensor.get_ptr();
  float* d_beta = weight_tensor.get_ptr() + num_feature;
  CK_CUDA_THROW_(
      cudaMemcpy(d_gamma, h_gamma.get(), num_feature * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(
      cudaMemcpy(d_beta, h_beta.get(), num_feature * sizeof(float), cudaMemcpyHostToDevice));

  simulator.fill(h_in.get(), len);

  batch_norm_fprop_cpu(h_gamma.get(), h_beta.get(), h_in.get(), h_expected.get(), batch_size,
                       num_feature);

  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  batch_norm_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, len * sizeof(float), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_expected.get(), len, eps));

  simulator.fill(h_out.get(), len);

  CK_CUDA_THROW_(cudaMemcpy(h_expected.get(), d_in, len * sizeof(float), cudaMemcpyDeviceToHost));
  batch_norm_bprop_cpu(h_gamma.get(), h_out.get(), h_expected.get(), batch_size, num_feature);

  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), len * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  batch_norm_layer.bprop();
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_in.get(), d_in, len * sizeof(float), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_expected.get(), len, eps));
}

}  // namespace

TEST(batch_norm_layer, fp32_2x4) { batch_norm_test(2, 4); }
TEST(batch_norm_layer, fp32_4x2) { batch_norm_test(4, 2); }
TEST(batch_norm_layer, fp32_1024x2) { batch_norm_test(1024, 2); }
TEST(batch_norm_layer, fp32_1024x511) { batch_norm_test(1024, 511); }
TEST(batch_norm_layer, fp32_1024x512) { batch_norm_test(1024, 512); }
TEST(batch_norm_layer, fp32_512x1024) { batch_norm_test(512, 1024); }
TEST(batch_norm_layer, fp32_511x1024) { batch_norm_test(511, 1024); }
