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

#include <cublas_v2.h>
#include <curand.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core23/low_level_primitives.hpp>
#include <cstdlib>
#include <layers/cast_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6;

void cast_cpu(__half* top, const float* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(bottom[i]);
  }
}

void cast_cpu(float* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __half2float(bottom[i]);
  }
}

template <typename From, typename To>
void cast_test(int64_t dim0, int64_t dim1) {
  constexpr bool use_mixed_precision = std::is_same_v<To, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .shape({dim0, dim1})
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());
  core23::Tensor bottom_tensor(tensor_params.data_type(
      use_mixed_precision ? core23::ScalarType::Float : core23::ScalarType::Half));
  core23::Tensor top_tensor(tensor_params);

  CastLayer<From, To> cast_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  cast_layer.initialize();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  const auto len = dim0 * dim1;

  std::vector<From> h_bottom(len);
  std::vector<To> h_top(len);
  std::vector<To> h_ref(len);

  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);

  // fprop test
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  cast_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  cast_cpu(h_ref.data(), h_bottom.data(), len);
  ASSERT_TRUE(test::compare_array_approx<To>(h_top.data(), h_ref.data(), len, eps));

  // bprop test
  // doing nothing in bprop, no need to test
  cast_layer.bprop();
}

TEST(cast_layer, fp32_fp16_32x64) { cast_test<float, __half>(32, 64); }
TEST(cast_layer, fp32_fp16_64x128) { cast_test<float, __half>(64, 128); }
TEST(cast_layer, fp32_fp16_128x256) { cast_test<float, __half>(128, 256); }
TEST(cast_layer, fp32_fp16_256x512) { cast_test<float, __half>(256, 512); }
TEST(cast_layer, fp16_fp32_32x64) { cast_test<__half, float>(32, 64); }
TEST(cast_layer, fp16_fp32_64x128) { cast_test<__half, float>(64, 128); }
TEST(cast_layer, fp16_fp32_128x256) { cast_test<__half, float>(128, 256); }
TEST(cast_layer, fp16_fp32_256x512) { cast_test<__half, float>(256, 512); }

}  // end namespace
