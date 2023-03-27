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

#include <gtest/gtest.h>

#include <HugeCTR/include/network_buffer_channels.hpp>
#include <core23/tensor_container.hpp>
#include <cstdlib>
#include <loss.hpp>
#include <regularizers/no_regularizer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
using namespace HugeCTR::test;

void transpose(float *a, int64_t m, int64_t n) {
  std::vector<float> tmp;
  tmp.resize(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) tmp[j * m + i] = a[i * n + j];
  for (int64_t i = 0; i < m * n; ++i) a[i] = tmp[i];
}
void cross_entropy_loss(int64_t batch_size) {
  int64_t feature_dim = 2;

  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  core23::Tensor input_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<float>::value)
                                                   .shape({batch_size, feature_dim})
                                                   .buffer_params(blobs_buffer_params));

  core23::Tensor label_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<float>::value)
                                                   .shape({batch_size, 1})
                                                   .buffer_params(blobs_buffer_params));

  core23::Tensor loss_tensor = core23::Tensor(core23::TensorParams()
                                                  .data_type(core23::ToScalarType<float>::value)
                                                  .shape({1, 1})
                                                  .buffer_params(blobs_buffer_params));

  core23::Tensor empty_tensor = core23::Tensor(core23::TensorParams()
                                                   .data_type(core23::ToScalarType<float>::value)
                                                   .shape({1, 1})
                                                   .buffer_params(blobs_buffer_params));

  WeightTensors weight_tensors({empty_tensor}, {1});
  WgradTensors<float> wgrad_tensors({empty_tensor}, {1});
  std::shared_ptr<NoRegularizer<float>> no_regularizer(
      new NoRegularizer<float>(weight_tensors, wgrad_tensors, batch_size, test::get_default_gpu()));

  CrossEntropyLoss<float> cel(label_tensor, input_tensor, loss_tensor, no_regularizer,
                              test::get_default_gpu(), 1);

  cel.set_label_weight(1.0);

  buff->allocate();

  float *d_input = input_tensor.data<float>();
  float *d_label = label_tensor.data<float>();
  float *d_loss = loss_tensor.data<float>();

  std::unique_ptr<float[]> h_input(new float[batch_size * feature_dim]);
  std::unique_ptr<float[]> h_label(new float[batch_size]);

  srand(time(NULL));
  for (int64_t i = 0; i < batch_size * feature_dim; ++i) h_input[i] = rand() % 100 * 0.01f;
  for (int64_t i = 0; i < batch_size; ++i) h_label[i] = rand() % 2;

  // GPU
  HCTR_LIB_THROW(cudaMemcpy(d_input, h_input.get(), sizeof(float) * batch_size * feature_dim,
                            cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_label, h_label.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  cel.compute_and_init(true);

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
  for (int64_t i = 0; i < batch_size; ++i) {
    z0_exp = exp(h_input[i * feature_dim]);
    z1_exp = exp(h_input[i * feature_dim + 1]);

    a0 = z0_exp / (z0_exp + z1_exp);
    a1 = z1_exp / (z0_exp + z1_exp);

    h_input[i * feature_dim] = (a0 - (h_label[i] == 0.0f ? 1 : 0)) / batch_size * scaler;
    h_input[i * feature_dim + 1] = (a1 - (h_label[i] == 1.0f ? 1 : 0)) / batch_size * scaler;

    cpu_loss += -1 * log(h_label[i] == 0.0f ? a0 : a1);
  }
  cpu_loss /= batch_size;

  ASSERT_EQ(true, cpu_gpu_cmp(&cpu_loss, d_loss, 1)) << " CSE Loss calulation failed" << std::endl;
  ASSERT_EQ(true, cpu_gpu_cmp(h_input.get(), d_input, batch_size * feature_dim))
      << " CSE Gradient calulation failed" << std::endl;
}

TEST(loss_test, CrossEntropyLoss_2048_row_major) { cross_entropy_loss(2048); }
TEST(loss_test, CrossEntropyLoss_64_row_major) { cross_entropy_loss(64); }

void binary_cross_entropy_loss(size_t batch_size) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  Tensor2<float> input_tensor;
  buff->reserve({batch_size, 1}, &input_tensor);
  Tensor2<float> label_tensor;
  buff->reserve({batch_size, 1}, &label_tensor);
  Tensor2<float> loss_tensor;
  buff->reserve({1, 1}, &loss_tensor);

  std::shared_ptr<BufferBlock2<float>> weight_buff = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = buff->create_block<float>();

  std::shared_ptr<NoRegularizer<float>> no_regularizer(new NoRegularizer<float>(
      weight_buff->as_tensor(), wgrad_buff->as_tensor(), batch_size, test::get_default_gpu()));

  BinaryCrossEntropyLoss<float> bce(label_tensor, input_tensor, loss_tensor, no_regularizer,
                                    test::get_default_gpu(), 1);
  bce.set_label_weight(1.0);

  buff->allocate();

  float *d_input = input_tensor.get_ptr();
  float *d_label = label_tensor.get_ptr();
  float *d_loss = loss_tensor.get_ptr();

  std::unique_ptr<float[]> h_input(new float[batch_size]);
  std::unique_ptr<float[]> h_label(new float[batch_size]);

  srand(time(NULL));
  for (int64_t i = 0; i < batch_size; ++i) h_input[i] = rand() % 100 * 0.01f;
  for (int64_t i = 0; i < batch_size; ++i) h_label[i] = rand() % 2;
  // GPU
  HCTR_LIB_THROW(
      cudaMemcpy(d_input, h_input.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(
      cudaMemcpy(d_label, h_label.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice));

  // Test with separate regularizer and compute methods
  float rterm = bce.regularizer_compute_rterm();
  const auto &input_dim = input_tensor.get_dimensions();
  bce.compute(true, input_dim[0], rterm);
  bce.regularizer_initialize_wgrad(true);

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
  for (int64_t i = 0; i < batch_size; ++i) {
    x = h_input[i];
    val = 1 / (1 + exp(-h_input[i]));
    y = h_label[i];

    h_input[i] = -1 * val * (y - val) * exp(-x) / (1 - val) / batch_size * scaler;
    cpu_loss += y * log(val) + (1 - y) * log(1 - val);
  }
  cpu_loss = -cpu_loss / batch_size;

  ASSERT_EQ(true, cpu_gpu_cmp(&cpu_loss, d_loss, 1)) << " CSE Loss calulation failed" << std::endl;
  ASSERT_EQ(true, cpu_gpu_cmp(h_input.get(), d_input, batch_size))
      << " CSE Gradient calulation failed" << std::endl;
}

TEST(loss_test, BinaryCrossEntropyLoss_2048) { binary_cross_entropy_loss(2048); }
TEST(loss_test, BinaryCrossEntropyLoss_64) { binary_cross_entropy_loss(64); }
