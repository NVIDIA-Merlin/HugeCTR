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

#include <layers/prelu_dice_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6;
template <typename T>
void transpose(T* B, T* A, int N, int M) {
  int i, j;
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) B[i * M + j] = A[j * N + i];
}
template <typename T>
void get_mean(T* E_x, T* in, const int len, int batchsize, int hiddensize) {
  for (int i = 0; i < hiddensize; i++) {
    E_x[i] = 0;
    int offset = i * batchsize;
    for (int j = 0; j < batchsize; j++) {
      E_x[i] += in[offset + j];
    }
    E_x[i] /= batchsize;
  }
}

template <typename T>
void get_variance(T* Var_x, T* E_x, T* in, const int len, int batchsize, int hiddensize) {
  T* x2 = new T[len];
  for (int i = 0; i < hiddensize; i++) {
    int offset = i * batchsize;
    for (int j = 0; j < batchsize; j++) {
      x2[offset + j] = pow(in[offset + j] - E_x[i], 2.0);
    }
  }
  get_mean(Var_x, x2, len, batchsize, hiddensize);
  delete[] x2;
}

template <typename T>
void dice_fprop(T* out, T* in, T* E_x, T* Var_x, int len, int batchsize, int hiddensize, T alpha,
                T epsilon) {
  for (int i = 0; i < hiddensize; i++) {
    for (int j = 0; j < batchsize; j++) {
      int index = i * batchsize + j;
      T ps = 1 / (1 + expf((E_x[i] - in[index]) / sqrt(Var_x[i] + epsilon)));
      out[index] = ps * in[index] + (1 - ps) * alpha * in[index];
    }
  }
}

template <typename T>
void dice_bprop(T* out, T* top, T* bottom, T* E_x, T* Var_x, int len, int N, int hiddensize,
                T alpha, T epsilon) {
  for (int i = 0; i < hiddensize; i++) {
    T Ex = E_x[i];
    T Vx = Var_x[i];
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      T s = bottom[index];
      T ys_s = (T(1.0 / N) - 1) * (1.0 / sqrt(Vx + epsilon)) -
               T(1.0 / N) * (Ex - s) * (1.0 / sqrt(pow(Vx + epsilon, 3.0))) * (s - Ex);
      T ys = (Ex - s) * (1.0 / sqrt(Vx + epsilon));

      T ps_s = -1.0 * expf(ys) * ys_s * (1.0 / pow(1.0 + expf(ys), 2.0));
      T ps = 1.0 / (1 + expf(ys));

      out[index] = ((ps_s * s + ps) * (1.0 - alpha) + alpha) * top[index];
    }
  }
}

template <typename T>
void prelu_dice_fprop_cpu(T* out, T* in, int len, int batchsize, T alpha, T epsilon) {
  int hiddensize = len / batchsize;
  T* E_x = new T[hiddensize];
  T* Var_x = new T[hiddensize];
  get_mean(E_x, in, len, batchsize, hiddensize);
  get_variance(Var_x, E_x, in, len, batchsize, hiddensize);
  dice_fprop(out, in, E_x, Var_x, len, batchsize, hiddensize, alpha, epsilon);
  delete[] E_x;
  delete[] Var_x;
}

template <typename T>
void prelu_dice_bprop_cpu(T* d_bottom, T* d_top, T* bottom, int len, int batchsize, T alpha,
                          T epsilon) {
  int hiddensize = len / batchsize;
  T* E_x = new T[hiddensize];
  T* Var_x = new T[hiddensize];
  get_mean(E_x, bottom, len, batchsize, hiddensize);
  get_variance(Var_x, E_x, bottom, len, batchsize, hiddensize);
  dice_bprop(d_bottom, d_top, bottom, E_x, Var_x, len, batchsize, hiddensize, alpha, epsilon);
  delete[] E_x;
  delete[] Var_x;
}

template <typename T>
void prelu_dice_test(int64_t batchsize, int64_t hiddensize) {
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  auto shape = core23::Shape({batchsize, hiddensize});

  core23::TensorParams tensor_params = core23::TensorParams(shape)
                                           .device(device)
                                           .data_type(core23::ScalarType::Float)
                                           .buffer_channel(core23::GetRandomBufferChannel());
  core23::Tensor input_tensor(tensor_params);
  core23::Tensor output_tensor(tensor_params);

  T alpha = 0.2;
  T epsilon = 1e-8;
  PRelu_Dice_Layer<T> prelu_dice_layer(input_tensor, output_tensor, alpha, epsilon,
                                       test::get_default_gpu());

  prelu_dice_layer.initialize();

  const int64_t len = hiddensize * batchsize;

  std::vector<T> h_bottom(len);
  std::vector<T> h_bottom_trans(len);
  std::vector<T> h_top(len);
  std::vector<T> h_top_trans(len);
  std::vector<T> d2h_top(len);
  std::vector<T> h_bottom_grad(len);
  std::vector<T> h_bottom_grad_trans(len);
  std::vector<T> d2h_bottom_grad(len);

  test::normal_sync_cpu(h_bottom.data(), len, 0.f, 1.f, generator);

  core23::copy_sync(input_tensor.data(), h_bottom.data(), input_tensor.num_bytes(),
                    input_tensor.device(), core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  prelu_dice_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_top.data(), output_tensor.data(), output_tensor.num_bytes(),
                    core23::DeviceType::CPU, output_tensor.device());

  transpose(h_bottom_trans.data(), h_bottom.data(), hiddensize, batchsize);
  prelu_dice_fprop_cpu<T>(h_top_trans.data(), h_bottom_trans.data(), len, batchsize, alpha,
                          epsilon);
  transpose(h_top.data(), h_top_trans.data(), batchsize, hiddensize);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.data(), h_top.data(), len, eps));
  // bprop
  test::normal_sync_cpu(h_top.data(), len, 0.f, 1.f, generator);
  core23::copy_sync(output_tensor.data(), h_top.data(), output_tensor.num_bytes(),
                    output_tensor.device(), core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  prelu_dice_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_bottom_grad.data(), input_tensor.data(), input_tensor.num_bytes(),
                    core23::DeviceType::CPU, input_tensor.device());
  transpose(h_top_trans.data(), h_top.data(), hiddensize, batchsize);
  prelu_dice_bprop_cpu<T>(h_bottom_grad_trans.data(), h_top_trans.data(), h_bottom_trans.data(),
                          len, batchsize, alpha, epsilon);
  transpose(h_bottom_grad.data(), h_bottom_grad_trans.data(), batchsize, hiddensize);
  ASSERT_TRUE(
      test::compare_array_approx<T>(d2h_bottom_grad.data(), h_bottom_grad.data(), len, eps));
}

}  // namespace
TEST(prelu_dice_layer, fp32_1x100) { prelu_dice_test<float>(1, 2); }
TEST(prelu_dice_layer, fp32_100x100) { prelu_dice_test<float>(100, 100); }
TEST(prelu_dice_layer, fp32_100x500) { prelu_dice_test<float>(20, 200); }   // 1,5; 100,500;
TEST(prelu_dice_layer, fp32_500x100) { prelu_dice_test<float>(500, 100); }  // 1,5; 100,500;
TEST(prelu_dice_layer, fp32_512x512) { prelu_dice_test<float>(512, 512); }
TEST(prelu_dice_layer, fp32_1048x512) { prelu_dice_test<float>(1024, 512); }
TEST(prelu_dice_layer, fp32_2048x512) { prelu_dice_test<float>(1024 * 2, 512); }
TEST(prelu_dice_layer, fp32_512x1048) { prelu_dice_test<float>(512, 1024); }
TEST(prelu_dice_layer, fp32_512x2048) { prelu_dice_test<float>(512, 1024 * 2); }
