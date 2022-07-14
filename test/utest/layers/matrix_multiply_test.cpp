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

#include <gtest/gtest.h>
#include <utest/test_utils.h>

#include <vector>

#include "HugeCTR/include/layers/matrix_multiply_layer.hpp"
#include "HugeCTR/include/utils.hpp"

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-3f; }
};

template <typename T>
void matmul_cpu(T *in1, T *in2, T *output, size_t B, size_t M, size_t N, size_t K) {
  size_t i, j, k, z;
  for (z = 0; z < B; z++) {
    for (i = 0; i < M; i++) {
      for (j = 0; j < K; j++) {
        output[z * M * K + i * K + j] = 0;
        for (k = 0; k < N; k++) {
          output[z * M * K + i * K + j] += in1[z * M * N + i * N + k] * in2[z * N * K + k * K + j];
        }
      }
    }
  }
}

template <typename T>
static void transpose(T *a, size_t b, size_t m, size_t n) {
  for (size_t z = 0; z < b; z++) {
    T *cur_a = a + z * m * n;
    std::unique_ptr<T[]> tmp(new T[m * n]);
    for (size_t i = 0; i < m; ++i)
      for (size_t j = 0; j < n; ++j) tmp[j * m + i] = cur_a[i * n + j];
    for (size_t i = 0; i < m * n; ++i) cur_a[i] = tmp[i];
  }
}

template <typename T>
void matmul_dgrad_cpu(T *out, T **h_ins, T **h_b_ins, size_t b, size_t m, size_t n, size_t k) {
  transpose(h_ins[1], b, n, k);
  transpose(h_ins[0], b, m, n);
  matmul_cpu(h_ins[0], out, h_b_ins[1], b, n, m, k);
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, m, k, n);
}

template <typename T>
void matmul_test(size_t b, size_t m, size_t n, size_t k) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  size_t out_size = b * m * k;
  int dims = 2;

  Tensors2<T> in_tensors;
  Tensor2<T> in_tensor;
  if (b == 1) {  // 2D inputs
    buff->reserve({m, n}, &in_tensor);
    in_tensors.push_back(in_tensor);
    buff->reserve({n, k}, &in_tensor);
    in_tensors.push_back(in_tensor);
  } else {  // 3D inputs
    buff->reserve({b, m, n}, &in_tensor);
    in_tensors.push_back(in_tensor);
    buff->reserve({b, n, k}, &in_tensor);
    in_tensors.push_back(in_tensor);
    dims = 3;
  }
  Tensor2<T> out_tensor;

  MatrixMultiplyLayer<T> matmul_layer(in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    size_t size =
        b * in_tensors[i].get_dimensions()[dims - 2] * in_tensors[i].get_dimensions()[dims - 1];
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    simulator.fill(h_ins[i], test::align_to_even(size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T *d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));
  matmul_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), b, m, n, k);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));

  // bprop
  simulator.fill(h_out.get(), test::align_to_even(out_size));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  matmul_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), b, m, n, k);
  for (size_t i = 0; i < num; i++) {
    size_t size =
        b * in_tensors[i].get_dimensions()[dims - 2] * in_tensors[i].get_dimensions()[dims - 1];
    T *d_out = in_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

template <typename T>
void matmul_test_4d(size_t b, size_t h, size_t m, size_t n, size_t k) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  size_t out_size = b * h * m * k;
  int dims = 4;

  Tensors2<T> in_tensors;
  Tensor2<T> in_tensor;
  // 4D inputs
  buff->reserve({b, h, m, n}, &in_tensor);
  in_tensors.push_back(in_tensor);
  buff->reserve({b, h, n, k}, &in_tensor);
  in_tensors.push_back(in_tensor);
  Tensor2<T> out_tensor;

  MatrixMultiplyLayer<T> matmul_layer(in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    size_t size = b * in_tensors[i].get_dimensions()[dims - 3] *
                  in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    simulator.fill(h_ins[i], test::align_to_even(size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T *d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));
  matmul_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), b * h, m, n, k);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));

  // bprop
  simulator.fill(h_out.get(), test::align_to_even(out_size));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  matmul_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), b * h, m, n, k);
  for (size_t i = 0; i < num; i++) {
    size_t size = b * in_tensors[i].get_dimensions()[dims - 3] *
                  in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    T *d_out = in_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

}  // namespace

// 2D inputs
TEST(matmul_layer, fp32_1x2x3x4) { matmul_test<float>(1, 2, 3, 4); }
TEST(matmul_layer, fp32_1x128x256x32) { matmul_test<float>(1, 128, 256, 32); }
TEST(matmul_layer, fp32_1x256x512x1024) { matmul_test<float>(1, 256, 512, 1024); }
TEST(matmul_layer, fp32_1x1024x512x256) { matmul_test<float>(1, 1024, 512, 256); }
TEST(matmul_layer, fp32_1x1024x2048x1024) { matmul_test<float>(1, 1024, 2048, 1024); }

// 3D inputs
TEST(matmul_layer, fp32_2x2x3x4) { matmul_test<float>(2, 2, 3, 4); }
TEST(matmul_layer, fp32_32x128x256x32) { matmul_test<float>(32, 128, 256, 32); }
TEST(matmul_layer, fp32_64x256x512x1024) { matmul_test<float>(64, 256, 512, 1024); }
TEST(matmul_layer, fp32_12x1024x512x256) { matmul_test<float>(12, 1024, 512, 256); }
TEST(matmul_layer, fp32_6x1024x2048x1024) { matmul_test<float>(6, 1024, 2048, 1024); }

// 4D inputs
TEST(matmul_layer, fp32_2x2x2x3x4) { matmul_test_4d<float>(2, 2, 2, 3, 4); }
TEST(matmul_layer, fp32_32x4x128x256x32) { matmul_test_4d<float>(32, 4, 128, 256, 32); }
TEST(matmul_layer, fp32_12x10x1024x512x256) { matmul_test_4d<float>(12, 4, 1024, 512, 256); }
TEST(matmul_layer, fp32_4x1x1024x2048x1024) { matmul_test_4d<float>(4, 1, 1024, 2048, 1024); }
