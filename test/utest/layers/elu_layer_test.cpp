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

#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/elu_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

template <typename T>
void elu_cpu(const T* in, T* out, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    out[i] = (in[i] < 0) ? T(alpha * (exp(in[i]) - 1)) : in[i];
  }
}

template <typename T>
void elu_bprop_cpu(const T* d_out, T* d_in, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    d_in[i] = (d_in[i] < 0) ? T(alpha * exp(d_in[i]) * d_out[i]) : d_out[i];
  }
}

template <typename T>
void elu_test(int64_t dim0, int64_t dim1, T alpha) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  auto shape = core23::Shape({dim0, dim1});

  core23::TensorParams tensor_params =
      core23::TensorParams(shape)
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());
  core23::Tensor bottom_tensor(tensor_params);
  core23::Tensor top_tensor(tensor_params);

  EluLayer<T> elu_layer(bottom_tensor, top_tensor, alpha, test::get_default_gpu());

  elu_layer.initialize();

  const int64_t len = dim0 * dim1;

  std::vector<T> h_bottom(len);
  std::vector<T> h_top(len);
  std::vector<T> h_expected(len);

  if constexpr (std::is_same_v<T, __half>) {
    std::vector<float> h_bottom_full(len);
    core23::normal_async<float>(h_bottom_full.data(), len, 0.f, 1.f, core23::DeviceType::CPU,
                                generator, stream);
    core23::convert_async<T, float>(h_bottom.data(), h_bottom_full.data(), len,
                                    core23::DeviceType::CPU, core23::DeviceType::CPU, stream);
  } else {
    core23::normal_async<T>(h_bottom.data(), len, 0.f, 1.f, core23::DeviceType::CPU, generator,
                            stream);
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  // fprop
  core23::copy_async(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                     bottom_tensor.device(), core23::DeviceType::CPU, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  elu_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  elu_cpu(h_bottom.data(), h_expected.data(), len, alpha);
  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_expected.data(), len, Eps<T>::value()));

  // bprop
  if constexpr (std::is_same_v<T, __half>) {
    std::vector<float> h_bottom_full(len);
    std::vector<float> h_top_full(len);
    core23::normal_async<float>(h_bottom_full.data(), len, 0.f, 1.f, core23::DeviceType::CPU,
                                generator, stream);
    core23::normal_async<float>(h_top_full.data(), len, 0.f, 1.f, core23::DeviceType::CPU,
                                generator, stream);
    core23::convert_async<T, float>(h_bottom.data(), h_bottom_full.data(), len,
                                    core23::DeviceType::CPU, core23::DeviceType::CPU, stream);
    core23::convert_async<T, float>(h_top.data(), h_top_full.data(), len, core23::DeviceType::CPU,
                                    core23::DeviceType::CPU, stream);
  } else {
    core23::normal_async<float>(h_bottom.data(), len, 0.f, 1.f, core23::DeviceType::CPU, generator,
                                stream);
    core23::normal_async<float>(h_top.data(), len, 0.f, 1.f, core23::DeviceType::CPU, generator,
                                stream);
  }
  cudaStreamSynchronize(stream());

  core23::copy_async(h_expected.data(), h_bottom.data(), h_bottom.size() * sizeof(T),
                     core23::DeviceType::CPU, core23::DeviceType::CPU, stream);

  core23::copy_async(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                     bottom_tensor.device(), core23::DeviceType::CPU, stream);
  core23::copy_async(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                     core23::DeviceType::CPU, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  elu_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(h_bottom.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());

  elu_bprop_cpu(h_top.data(), h_expected.data(), len, alpha);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_bottom.data(), h_expected.data(), len, Eps<T>::value()));
}

}  // namespace

TEST(elu_layer, fp32_10x20_1) { elu_test<float>(10, 20, 1.0); }
TEST(elu_layer, fp32_10x500_1) { elu_test<float>(10, 500, 1.0); }
TEST(elu_layer, fp32_512x2048_1) { elu_test<float>(512, 1024 * 2, 1.0); }

TEST(elu_layer, fp16_10x20_1) { elu_test<__half>(10, 20, 1.0); }
TEST(elu_layer, fp16_10x500_1) { elu_test<__half>(10, 500, 1.0); }
TEST(elu_layer, fp16_512x2048_1) { elu_test<__half>(512, 1024 * 2, 1.0); }

TEST(elu_layer, fp16_9x19_1) { elu_test<__half>(10, 20, 1.0); }
TEST(elu_layer, fp16_9x499_1) { elu_test<__half>(10, 500, 1.0); }
TEST(elu_layer, fp16_511x2047_1) { elu_test<__half>(512, 1024 * 2, 1.0); }
