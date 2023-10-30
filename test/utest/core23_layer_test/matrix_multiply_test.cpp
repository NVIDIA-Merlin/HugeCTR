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

#include <layers/matrix_multiply_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

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
void matmul_test(int64_t b, int64_t m, int64_t n, int64_t k) {
  auto out_size = b * m * k;
  int dims = 2;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params = core23::TensorParams()
                                           .device(device)
                                           .data_type(core23::ToScalarType<T>::value)
                                           .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> input_tensors;
  core23::Tensor output_tensor;

  if (b == 1) {  // 2D inputs
    input_tensors.push_back(core23::Tensor(tensor_params.shape({m, n})));
    input_tensors.push_back(core23::Tensor(tensor_params.shape({n, k})));
  } else {  // 3D inputs
    input_tensors.push_back(core23::Tensor(tensor_params.shape({b, m, n})));
    input_tensors.push_back(core23::Tensor(tensor_params.shape({b, n, k})));
    dims = 3;
  }

  MatrixMultiplyLayer<T> matmul_layer(input_tensors, output_tensor, test::get_default_gpu());

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = input_tensors[i].data<T>();
  }

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // fprop
  for (size_t i = 0; i < num; i++) {
    auto size =
        b * input_tensors[i].shape().size(dims - 2) * input_tensors[i].shape().size(dims - 1);
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    test::normal_sync_cpu(h_ins[i], test::align_to_even(size), 0.f, 1.f, generator);
    core23::copy_sync(h_d_ins[i], h_ins[i], size * sizeof(T), input_tensors[i].device(),
                      core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T *d_out = output_tensor.data<T>();
  core23::copy_sync(h_out.get(), d_out, out_size * sizeof(T), core23::DeviceType::CPU,
                    output_tensor.device());
  matmul_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), b, m, n, k);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));

  // bprop
  test::normal_sync_cpu(h_out.get(), test::align_to_even(out_size), 0.f, 1.f, generator);
  core23::copy_sync(d_out, h_out.get(), out_size * sizeof(T), output_tensor.device(),
                    core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  matmul_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), b, m, n, k);
  for (size_t i = 0; i < num; i++) {
    auto size =
        b * input_tensors[i].shape().size(dims - 2) * input_tensors[i].shape().size(dims - 1);
    T *d_out = input_tensors[i].data<T>();
    core23::copy_sync(h_bprop_out[i], d_out, size * sizeof(T), core23::DeviceType::CPU,
                      input_tensors[i].device());
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

template <typename T>
void matmul_test_4d(int64_t b, int64_t h, int64_t m, int64_t n, int64_t k) {
  auto out_size = b * h * m * k;
  int dims = 4;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params = core23::TensorParams()
                                           .device(device)
                                           .data_type(core23::ToScalarType<T>::value)
                                           .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> input_tensors;
  core23::Tensor output_tensor;

  // 4D inputs
  input_tensors.push_back(core23::Tensor(tensor_params.shape({b, h, m, n})));
  input_tensors.push_back(core23::Tensor(tensor_params.shape({b, h, n, k})));

  MatrixMultiplyLayer<T> matmul_layer(input_tensors, output_tensor, test::get_default_gpu());

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = input_tensors[i].data<T>();
  }

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  // fprop
  for (size_t i = 0; i < num; i++) {
    auto size = b * input_tensors[i].shape().size(dims - 3) *
                input_tensors[i].shape().size(dims - 2) * input_tensors[i].shape().size(dims - 1);
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    test::normal_sync_cpu(h_ins[i], test::align_to_even(size), 0.f, 1.f, generator);
    core23::copy_sync(h_d_ins[i], h_ins[i], size * sizeof(T), input_tensors[i].device(),
                      core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T *d_out = output_tensor.data<T>();
  core23::copy_sync(h_out.get(), d_out, out_size * sizeof(T), core23::DeviceType::CPU,
                    output_tensor.device());
  matmul_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), b * h, m, n, k);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size, Eps<T>::value()));

  // bprop
  test::normal_sync_cpu(h_out.get(), test::align_to_even(out_size), 0.f, 1.f, generator);
  core23::copy_sync(d_out, h_out.get(), out_size * sizeof(T), output_tensor.device(),
                    core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  matmul_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  matmul_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), b * h, m, n, k);
  for (size_t i = 0; i < num; i++) {
    auto size = b * input_tensors[i].shape().size(dims - 3) *
                input_tensors[i].shape().size(dims - 2) * input_tensors[i].shape().size(dims - 1);
    T *d_out = input_tensors[i].data<T>();
    core23::copy_sync(h_bprop_out[i], d_out, size * sizeof(T), core23::DeviceType::CPU,
                      input_tensors[i].device());
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
TEST(matmul_layer, fp32_1x256x512x256) { matmul_test<float>(1, 256, 512, 256); }

// 3D inputs
TEST(matmul_layer, fp32_2x2x3x4) { matmul_test<float>(2, 2, 3, 4); }
TEST(matmul_layer, fp32_32x128x256x32) { matmul_test<float>(32, 128, 256, 32); }
TEST(matmul_layer, fp32_6x256x512x256) { matmul_test<float>(6, 256, 512, 256); }
TEST(matmul_layer, fp32_12x1024x512x256) { matmul_test<float>(12, 1024, 512, 256); }

// 4D inputs
TEST(matmul_layer, fp32_2x2x2x3x4) { matmul_test_4d<float>(2, 2, 2, 3, 4); }
TEST(matmul_layer, fp32_32x4x128x256x32) { matmul_test_4d<float>(32, 4, 128, 256, 32); }
TEST(matmul_layer, fp32_3x4x512x256x128) { matmul_test_4d<float>(3, 4, 512, 256, 128); }
TEST(matmul_layer, fp32_2x1x512x512x512) { matmul_test_4d<float>(2, 1, 512, 512, 512); }
