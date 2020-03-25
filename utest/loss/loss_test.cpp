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

#include "HugeCTR/include/loss.hpp"
#include <cstdlib>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
using namespace std;
using namespace HugeCTR;
using namespace HugeCTR::test;

void transpose(float *a, int m, int n) {
  std::vector<float> tmp;
  tmp.resize(m*n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) tmp[j * m + i] = a[i * n + j];
  for (int i = 0; i < m * n; ++i) a[i] = tmp[i];
}
void cross_entropy_loss(int batch_size, bool row_major) {
  int feature_dim = 2;

  std::shared_ptr<GeneralBuffer<float>> input_b(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> label_b(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> loss_b(new GeneralBuffer<float>());

  std::shared_ptr<Tensor<float>> input_tensor(
      new Tensor<float>(row_major ? std::vector<int>{batch_size, feature_dim}
                                  : std::vector<int>{feature_dim, batch_size},
                        input_b, row_major ? TensorFormat_t::HW : TensorFormat_t::WH));
  std::shared_ptr<Tensor<float>> label_tensor(new Tensor<float>(
      row_major ? std::vector<int>{batch_size, 1} : std::vector<int>{1, batch_size}, label_b,
      row_major ? TensorFormat_t::HW : TensorFormat_t::WH));
  std::shared_ptr<Tensor<float>> loss_tensor(new Tensor<float>({1, 1}, loss_b, TensorFormat_t::HW));

  CrossEntropyLoss cel(label_tensor, input_tensor, loss_tensor, nullptr, 0);

  input_b->init(0);
  label_b->init(0);
  loss_b->init(0);

  float *d_input = input_b->get_ptr_with_offset(0);
  float *d_label = label_b->get_ptr_with_offset(0);
  float *d_loss = loss_b->get_ptr_with_offset(0);

  std::unique_ptr<float[]> h_input(new float[batch_size * feature_dim]);
  std::unique_ptr<float[]> h_label(new float[batch_size]);

  srand(time(NULL));
  for (int i = 0; i < batch_size * feature_dim; ++i) h_input[i] = rand() % 100 * 0.01f;
  for (int i = 0; i < batch_size; ++i) h_label[i] = rand() % 2;

  // GPU
  if (!row_major) transpose(h_input.get(), batch_size, feature_dim);
  cudaMemcpy(d_input, h_input.get(), sizeof(float) * batch_size * feature_dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_label, h_label.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice);
  cel.fused_loss_computation(cudaStreamDefault);

  if (!row_major) transpose(h_input.get(), feature_dim, batch_size);
  // CPU
  float z0_exp, z1_exp;
  float a0, a1;
  float cpu_loss = 0.0f;
  int scaler = 1;
#ifdef SCALE_128
  scaler = 128;
#elif SCALE_256
  scaler = 256;
#elif SCALE_512
  scaler = 512;
#elif SCALE_1024
  scaler = 1024;
#endif
  for (int i = 0; i < batch_size; ++i) {
    z0_exp = exp(h_input[i * feature_dim]);
    z1_exp = exp(h_input[i * feature_dim + 1]);

    a0 = z0_exp / (z0_exp + z1_exp);
    a1 = z1_exp / (z0_exp + z1_exp);

    h_input[i * feature_dim] = (a0 - (h_label[i] == 0.0f ? 1 : 0)) / batch_size * scaler;
    h_input[i * feature_dim + 1] = (a1 - (h_label[i] == 1.0f ? 1 : 0)) / batch_size * scaler;

    cpu_loss += -1 * log(h_label[i] == 0.0f ? a0 : a1);
  }
  cpu_loss /= batch_size;

  ASSERT_EQ(true, cpu_gpu_cmp(&cpu_loss, d_loss, 1)) << " CSE Loss calulation failed" << endl;
  if (!row_major) transpose(h_input.get(), batch_size, feature_dim);
  ASSERT_EQ(true, cpu_gpu_cmp(h_input.get(), d_input, batch_size * feature_dim))
      << " CSE Gradient calulation failed" << endl;
}

TEST(loss_test, CrossEntropyLoss_2048_row_major) { cross_entropy_loss(2048, true); }
TEST(loss_test, CrossEntropyLoss_2048_col_major) { cross_entropy_loss(2048, false); }
TEST(loss_test, CrossEntropyLoss_64_row_major) { cross_entropy_loss(64, true); }
TEST(loss_test, CrossEntropyLoss_64_col_major) { cross_entropy_loss(64, false); }

void binary_cross_entropy_loss(int batch_size, bool row_major) {
  std::shared_ptr<GeneralBuffer<float>> input_b(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> label_b(new GeneralBuffer<float>());
  std::shared_ptr<GeneralBuffer<float>> loss_b(new GeneralBuffer<float>());

  std::shared_ptr<Tensor<float>> input_tensor(new Tensor<float>(
      row_major ? std::vector<int>{batch_size, 1} : std::vector<int>{1, batch_size}, input_b,
      row_major ? TensorFormat_t::HW : TensorFormat_t::WH));
  std::shared_ptr<Tensor<float>> label_tensor(new Tensor<float>(
      row_major ? std::vector<int>{batch_size, 1} : std::vector<int>{1, batch_size}, label_b,
      row_major ? TensorFormat_t::HW : TensorFormat_t::WH));
  std::shared_ptr<Tensor<float>> loss_tensor(new Tensor<float>({1, 1}, loss_b, TensorFormat_t::HW));

  BinaryCrossEntropyLoss bce(label_tensor, input_tensor, loss_tensor, nullptr, 0);

  input_b->init(0);
  label_b->init(0);
  loss_b->init(0);

  float *d_input = input_b->get_ptr_with_offset(0);
  float *d_label = label_b->get_ptr_with_offset(0);
  float *d_loss = loss_b->get_ptr_with_offset(0);

  std::unique_ptr<float[]> h_input(new float[batch_size]);
  std::unique_ptr<float[]> h_label(new float[batch_size]);

  srand(time(NULL));
  for (int i = 0; i < batch_size; ++i) h_input[i] = rand() % 100 * 0.01f;
  for (int i = 0; i < batch_size; ++i) h_label[i] = rand() % 2;
  // GPU
  cudaMemcpy(d_input, h_input.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_label, h_label.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice);
  bce.fused_loss_computation(cudaStreamDefault);

  float cpu_loss = 0.0f;
  float x, val, y;
  int scaler = 1;
#ifdef SCALE_128
  scaler = 128;
#elif SCALE_256
  scaler = 256;
#elif SCALE_512
  scaler = 512;
#elif SCALE_1024
  scaler = 1024;
#endif
  for (int i = 0; i < batch_size; ++i) {
    x = h_input[i];
    val = 1 / (1 + exp(-h_input[i]));
    y = h_label[i];

    h_input[i] = -1 * val * (y - val) * exp(-x) / (1 - val) / batch_size * scaler;
    cpu_loss += y * log(val) + (1 - y) * log(1 - val);
  }
  cpu_loss = -cpu_loss / batch_size;

  ASSERT_EQ(true, cpu_gpu_cmp(&cpu_loss, d_loss, 1)) << " CSE Loss calulation failed" << endl;
  ASSERT_EQ(true, cpu_gpu_cmp(h_input.get(), d_input, batch_size))
      << " CSE Gradient calulation failed" << endl;
}

TEST(loss_test, BinaryCrossEntropyLoss_2048_row_major) { binary_cross_entropy_loss(2048, true); }
TEST(loss_test, BinaryCrossEntropyLoss_64_row_major) { binary_cross_entropy_loss(64, true); }
TEST(loss_test, BinaryCrossEntropyLoss_2048_col_major) { binary_cross_entropy_loss(2048, false); }
TEST(loss_test, BinaryCrossEntropyLoss_64_col_major) { binary_cross_entropy_loss(64, false); }
