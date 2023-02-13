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

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <functional>
#include <layers/sigmoid_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-2;

template <typename T>
void sigmoid_cpu(T* top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = T(1.) / (T(1.) + exp(-bottom[i]));
  }
}

template <>
void sigmoid_cpu(__half* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(1.0 / (1.0 + exp(-__half2float(bottom[i]))));
  }
}

template <typename T>
void sigmoid_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    T y = T(1.) / (T(1.) + exp(-bottom[i]));
    d_bottom[i] = d_top[i] * y * (T(1.) - y);
  }
}

template <>
void sigmoid_bprop_cpu(__half* d_bottom, const __half* d_top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    float y = 1.0 / (1.0 + exp(-__half2float(bottom[i])));
    d_bottom[i] = __float2half(__half2float(d_top[i]) * y * (1.0 - y));
  }
}

template <typename T>
void sigmoid_test(int64_t dim0, int64_t dim1) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::Shape shape = {dim0, dim1};

  core23::TensorParams tensor_params =
      core23::TensorParams(shape)
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::Tensor bottom_tensor(tensor_params);
  core23::Tensor top_tensor(tensor_params);

  SigmoidLayer<T> sigmoid_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  sigmoid_layer.initialize();

  const auto len = shape.size();

  std::vector<T> h_bottom(len);
  std::vector<T> h_top(len);
  std::vector<T> d2h_top(len);
  std::vector<T> h_bottom_grad(len);
  std::vector<T> d2h_bottom_grad(len);

  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  // fprop
  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sigmoid_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  sigmoid_cpu<T>(h_top.data(), h_bottom.data(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.data(), h_top.data(), len, eps));

  // bprop
  test::normal_sync_cpu(h_top.data(), h_top.size(), 0.f, 1.f, generator);

  core23::copy_sync(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                    core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sigmoid_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_bottom_grad.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());

  sigmoid_bprop_cpu<T>(h_bottom_grad.data(), h_top.data(), h_bottom.data(), len);
  ASSERT_TRUE(
      test::compare_array_approx<T>(d2h_bottom_grad.data(), h_bottom_grad.data(), len, eps));
}

}  // namespace

TEST(sigmoid_layer, fp32_10x20) { sigmoid_test<float>(10, 20); }
TEST(sigmoid_layer, fp32_10x500) { sigmoid_test<float>(10, 500); }
TEST(sigmoid_layer, fp32_512x2048) { sigmoid_test<float>(512, 1024 * 2); }
TEST(sigmoid_layer, fp16_10x20) { sigmoid_test<__half>(10, 20); }
TEST(sigmoid_layer, fp16_10x500) { sigmoid_test<__half>(10, 500); }
TEST(sigmoid_layer, fp16_512x2048) { sigmoid_test<__half>(512, 1024 * 2); }
