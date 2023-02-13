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
#include <gtest/gtest.h>

#include <cmath>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <cstdlib>
#include <layers/dropout_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6;
const float thr = 0.95f;

template <typename T>
void dropout_test(int64_t dim0, int64_t dim1, float rate) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  auto shape = core23::Shape({dim0, dim1});
  auto len = shape.size();

  core23::TensorParams tensor_params =
      core23::TensorParams(shape)
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());
  core23::Tensor bottom_tensor(tensor_params);
  core23::Tensor top_tensor(tensor_params);

  DropoutLayer<T> dropout_layer(bottom_tensor, top_tensor, rate, test::get_default_gpu());

  std::vector<T> h_bottom(len);
  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);

  // fprop test
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  dropout_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<T> h_top(len);
  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  int cnt_zero_fprop = 0;
  for (int i = 0; i < len; i++) {
    if (std::abs(h_top[i] - 0.f) < eps) {
      cnt_zero_fprop++;
    }
  }

  float ref_zero_cnt = rate * len;
  float p = (cnt_zero_fprop < ref_zero_cnt) ? ref_zero_cnt : cnt_zero_fprop;
  float c = (cnt_zero_fprop < ref_zero_cnt) ? cnt_zero_fprop : ref_zero_cnt;
  ASSERT_TRUE(c / p > thr);

  // bprop test
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  dropout_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_bottom.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());
  int cnt_zero_bprop = 0;
  for (int i = 0; i < len; i++) {
    if (std::abs(h_bottom[i] - 0.f) < eps) {
      cnt_zero_bprop++;
    }
  }
  ref_zero_cnt = rate * len;
  p = (cnt_zero_bprop < ref_zero_cnt) ? ref_zero_cnt : cnt_zero_bprop;
  c = (cnt_zero_bprop < ref_zero_cnt) ? cnt_zero_bprop : ref_zero_cnt;
  ASSERT_TRUE(c / p > thr);
}

TEST(dropout_layer, fp32_2048x1024_25) { dropout_test<float>(2048, 1024, 0.25); }

TEST(dropout_layer, fp32_2048x1024_50) { dropout_test<float>(2048, 1024, 0.50); }

TEST(dropout_layer, fp32_2048x1024_75) { dropout_test<float>(2048, 1024, 0.75); }

TEST(dropout_layer, fp32_2048x1024_99) { dropout_test<float>(2048, 1024, 0.99); }

TEST(dropout_layer, fp16_2048x1024_25) { dropout_test<__half>(2048, 1024, 0.25); }

TEST(dropout_layer, fp16_2048x1024_50) { dropout_test<__half>(2048, 1024, 0.50); }

TEST(dropout_layer, fp16_2048x1024_75) { dropout_test<__half>(2048, 1024, 0.75); }

TEST(dropout_layer, fp16_2048x1024_99) { dropout_test<__half>(2048, 1024, 0.99); }

}  // end namespace
