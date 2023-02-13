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
#include <layers/relu_layer.hpp>
#include <network_buffer_channels.hpp>
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
  static __half value() { return __float2half(1e-6f); }
};

template <typename T>
void relu_cpu(T* top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > T(0.)) {
      top[i] = bottom[i];
    } else {
      top[i] = T(0.);
    }
  }
}

template <typename T>
void relu_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > T(0.)) {
      d_bottom[i] = d_top[i];
    } else {
      d_bottom[i] = T(0.);
    }
  }
}

template <typename T>
void relu_test(int64_t dim0, int64_t dim1) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  auto shape = core23::Shape({dim0, dim1});

  core23::TensorParams tensor_params =
      core23::TensorParams(shape)
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(GetBlobsBufferChannel());
  core23::Tensor bottom_tensor(tensor_params);
  core23::Tensor top_tensor(tensor_params);

  ReluLayer<T> relu_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  relu_layer.initialize();

  const int64_t len = dim0 * dim1;

  std::vector<T> h_bottom(len);
  std::vector<T> h_top(len);
  std::vector<T> d2h_top(len);
  std::vector<T> h_bottom_grad(len);
  std::vector<T> d2h_bottom_grad(len);

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
  relu_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  relu_cpu<T>(h_top.data(), h_bottom.data(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.data(), h_top.data(), len, Eps<T>::value()));

  // bprop
  if constexpr (std::is_same_v<T, __half>) {
    std::vector<float> h_top_full(len);
    core23::normal_async<float>(h_top_full.data(), len, 0.f, 1.f, core23::DeviceType::CPU,
                                generator, stream);
    core23::convert_async<T, float>(h_top.data(), h_top_full.data(), len, core23::DeviceType::CPU,
                                    core23::DeviceType::CPU, stream);
  } else {
    core23::normal_async<float>(h_top.data(), len, 0.f, 1.f, core23::DeviceType::CPU, generator,
                                stream);
  }
  cudaStreamSynchronize(stream());

  core23::copy_async(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                     core23::DeviceType::CPU, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  relu_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_bottom_grad.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  relu_bprop_cpu<T>(h_bottom_grad.data(), h_top.data(), h_bottom.data(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.data(), h_bottom_grad.data(), len,
                                            Eps<T>::value()));
}

}  // namespace

TEST(relu_layer, fp32_10x20) { relu_test<float>(10, 20); }
TEST(relu_layer, fp32_10x500) { relu_test<float>(10, 500); }
TEST(relu_layer, fp32_512x2048) { relu_test<float>(512, 1024 * 2); }

TEST(relu_layer, fp16_10x20) { relu_test<__half>(10, 20); }
TEST(relu_layer, fp16_10x500) { relu_test<__half>(10, 500); }
TEST(relu_layer, fp16_512x2048) { relu_test<__half>(512, 1024 * 2); }

TEST(relu_layer, fp16_9x19) { relu_test<__half>(9, 19); }
TEST(relu_layer, fp16_9x499) { relu_test<__half>(9, 499); }
TEST(relu_layer, fp16_511x2047) { relu_test<__half>(511, 2047); }
