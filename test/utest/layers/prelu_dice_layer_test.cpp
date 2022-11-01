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

#include "HugeCTR/include/layers/prelu_dice_layer.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

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
void prelu_dice_test(size_t batchsize, size_t hiddensize) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  std::vector<size_t> dims = {batchsize, hiddensize};

  Tensor2<T> in_tensor;
  buf->reserve(dims, &in_tensor);
  Tensor2<T> out_tensor;
  buf->reserve(dims, &out_tensor);

  T alpha = 0.2;
  T epsilon = 1e-8;
  PRelu_Dice_Layer<T> prelu_dice_layer(in_tensor, out_tensor, buf, alpha, epsilon,
                                       test::get_default_gpu());

  buf->allocate();
  prelu_dice_layer.initialize();

  const size_t len = hiddensize * batchsize;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> h_bottom_trans(new T[len]);
  std::unique_ptr<T[]> h_top(new T[len]);
  std::unique_ptr<T[]> h_top_trans(new T[len]);
  std::unique_ptr<T[]> d2h_top(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad_trans(new T[len]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_bottom.get(), len);

  HCTR_LIB_THROW(
      cudaMemcpy(in_tensor.get_ptr(), h_bottom.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  prelu_dice_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_top.get(), out_tensor.get_ptr(), len * sizeof(T), cudaMemcpyDeviceToHost));

  transpose(h_bottom_trans.get(), h_bottom.get(), hiddensize, batchsize);
  prelu_dice_fprop_cpu<T>(h_top_trans.get(), h_bottom_trans.get(), len, batchsize, alpha, epsilon);
  transpose(h_top.get(), h_top_trans.get(), batchsize, hiddensize);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), len, eps));
  // bprop
  simulator.fill(h_top.get(), len);
  HCTR_LIB_THROW(
      cudaMemcpy(out_tensor.get_ptr(), h_top.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  prelu_dice_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(d2h_bottom_grad.get(), in_tensor.get_ptr(), len * sizeof(T),
                            cudaMemcpyDeviceToHost));
  transpose(h_top_trans.get(), h_top.get(), hiddensize, batchsize);
  prelu_dice_bprop_cpu<T>(h_bottom_grad_trans.get(), h_top_trans.get(), h_bottom_trans.get(), len,
                          batchsize, alpha, epsilon);
  transpose(h_bottom_grad.get(), h_bottom_grad_trans.get(), batchsize, hiddensize);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, eps));
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
