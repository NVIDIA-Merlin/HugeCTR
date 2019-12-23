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

#include "HugeCTR/include/layers/batch_norm_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <cudnn.h>

#include <math.h>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5;

void batch_norm_fprop_cpu(const float* gamma, const float* beta, const float* in, float* out,
                          bool row_major, int batch_size, int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      mean += in[idx];
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      float in_norm = (in[idx] - mean) / sqrt(var + eps);
      out[idx] = gamma[j] * in_norm + beta[j];
    }
  }
}

void batch_norm_bprop_cpu(const float* gamma, const float* out, float* in, bool row_major,
                          int batch_size, int num_feature) {
  for (int j = 0; j < num_feature; j++) {
    float mean = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      mean += in[idx];
    }
    mean /= batch_size;

    float var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      float diff = in[idx] - mean;
      var += (diff * diff);
    }
    var /= batch_size;

    float inv_std = 1.0f / sqrt(var + eps);

    float d_var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      float val = (out[idx] * gamma[j]) * (in[idx] - mean);
      d_var += val;
    }
    d_var *= (-0.5f) * pow(inv_std, 3);

    float val1 = 0.0f;
    float val2 = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      val1 += (out[idx] * gamma[j]);
      val2 += (in[idx] - mean);
    }
    val1 *= (-inv_std);
    val2 *= (d_var / batch_size) * -2;
    float d_mean = (val1 + val2);

    for (int i = 0; i < batch_size; i++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      in[idx] = (out[idx] * gamma[j]) * inv_std + d_var * (2.0 / batch_size) * (in[idx] - mean) +
                d_mean / batch_size;
    }
  }
}

void batch_norm_test(bool row_major, int batch_size, int num_feature) {
  std::shared_ptr<GeneralBuffer<float>> wbuff(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> wgbuff(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> blobs(new GeneralBuffer<float>());

  vector<int> dims = {row_major ? batch_size : num_feature, row_major ? num_feature : batch_size};
  TensorFormat_t format = row_major ? TensorFormat_t::HW : TensorFormat_t::WH;

  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(dims, blobs, format));
  std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(dims, blobs, format));

  cudnnHandle_t cudnn_handle;
  CK_CUDNN_THROW_(cudnnCreate(&cudnn_handle));

  BatchNormLayer::Params params = {true, 1.0, eps};
  BatchNormLayer batch_norm_layer(wbuff, wgbuff, in_tensor, out_tensor, params, cudnn_handle, 0);

  wbuff->init(0);
  wgbuff->init(0);
  blobs->init(0);

  const int len = batch_size * num_feature;

  float* d_in = in_tensor->get_ptr();
  float* d_out = out_tensor->get_ptr();

  std::unique_ptr<float[]> h_gamma(new float[num_feature]);
  std::unique_ptr<float[]> h_beta(new float[num_feature]);
  std::unique_ptr<float[]> h_in(new float[len]);
  std::unique_ptr<float[]> h_out(new float[len]);
  std::unique_ptr<float[]> h_expected(new float[len]);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -100.0, 100.0);

  // standard normall distribution is assumed
  for (int j = 0; j < num_feature; j++) {
    h_gamma[j] = 1.0;
    h_beta[j] = 0.0;
  }

  float* d_gamma = wbuff->get_ptr_with_offset(0);
  float* d_beta = wbuff->get_ptr_with_offset(num_feature);
  cudaMemcpy(d_gamma, h_gamma.get(), num_feature * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta.get(), num_feature * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_feature; j++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      h_in[idx] = simulator.get_num();
    }
  }

  batch_norm_fprop_cpu(h_gamma.get(), h_beta.get(), h_in.get(), h_expected.get(), row_major,
                       batch_size, num_feature);

  cudaMemcpy(d_in, h_in.get(), len * sizeof(float), cudaMemcpyHostToDevice);
  batch_norm_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, len * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_expected.get(), len, eps));

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_feature; j++) {
      int idx = row_major ? i * num_feature + j : j * batch_size + i;
      h_out[idx] = simulator.get_num();
    }
  }

  cudaMemcpy(h_expected.get(), d_in, len * sizeof(float), cudaMemcpyDeviceToHost);
  batch_norm_bprop_cpu(h_gamma.get(), h_out.get(), h_expected.get(), row_major, batch_size,
                       num_feature);

  cudaMemcpy(d_out, h_out.get(), len * sizeof(float), cudaMemcpyHostToDevice);
  batch_norm_layer.bprop(cudaStreamDefault);
  cudaMemcpy(h_in.get(), d_in, len * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_expected.get(), len, eps));

  CK_CUDNN_THROW_(cudnnDestroy(cudnn_handle));
}

}  // namespace

TEST(batch_norm_layer, fprop_and_bprop_WH) {
  batch_norm_test(false, 2, 4);
  batch_norm_test(false, 4, 2);
  batch_norm_test(false, 1024, 2);
  batch_norm_test(false, 1024, 511);
  batch_norm_test(false, 1024, 512);
  batch_norm_test(false, 512, 1024);
  batch_norm_test(false, 511, 1024);
}

TEST(batch_norm_layer, fprop_and_bprop_HW) {
  batch_norm_test(true, 2, 4);
  batch_norm_test(true, 4, 2);
  batch_norm_test(true, 1024, 2);
  batch_norm_test(true, 1024, 511);
  batch_norm_test(true, 1024, 512);
  batch_norm_test(true, 512, 1024);
  batch_norm_test(true, 511, 1024);
}
