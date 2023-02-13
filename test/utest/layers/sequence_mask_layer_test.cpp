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
#include <layers/sequence_mask_layer.hpp>
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
  static constexpr float value() { return 1e-6f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

template <typename T>
void f2i_input(T* input, size_t in_size, size_t max_sequence_len) {
  for (size_t i = 0; i < in_size; i++) {
    input[i] = abs(floor(input[i] * max_sequence_len));
  }
}

template <typename T>
void sequence_mask_cpu(T* input, T* output, size_t batch_size, size_t max_sequence_len,
                       size_t out_size) {
  for (size_t i = 0; i < batch_size; i++) {
    float length = input[i];
    for (size_t j = 0; j < max_sequence_len; j++) {
      if (j < length) {
        output[i * max_sequence_len + j] = (T)(1.0f);
      } else {
        output[i * max_sequence_len + j] = (T)(0.0f);
      }
    }
  }
}

template <typename T>
void sequence_mask_test(int64_t batch_size, int64_t max_sequence_len) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::Shape shape_in = {batch_size};
  core23::Shape shape_out = {batch_size, 1, 1, max_sequence_len};
  auto in_size = shape_in.size();
  auto out_size = shape_out.size();

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::Tensor bottom_tensor(tensor_params.shape(shape_in));
  core23::Tensor top_tensor(tensor_params.shape(shape_out));

  SequenceMaskLayer<T> sequence_mask_layer(bottom_tensor, top_tensor, max_sequence_len,
                                           test::get_default_gpu());

  auto* d_bottom = bottom_tensor.data<T>();
  auto* d_top = top_tensor.data<T>();

  std::vector<T> h_bottom(in_size);
  std::vector<T> h_top(out_size);
  std::vector<T> h_ref(out_size);

  // fprop
  test::normal_sync_cpu(h_bottom.data(), in_size, 0.f, 1.f, generator);

  f2i_input(h_bottom.data(), in_size, max_sequence_len);
  core23::copy_sync(d_bottom, h_bottom.data(), bottom_tensor.num_bytes(), bottom_tensor.device(),
                    core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sequence_mask_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), d_top, top_tensor.num_bytes(), core23::DeviceType::CPU,
                    top_tensor.device());

  sequence_mask_cpu(h_bottom.data(), h_ref.data(), batch_size, max_sequence_len, out_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_ref.data(), out_size, Eps<T>::value()));
}

}  // namespace

TEST(sequence_mask_layer, fp32_8192x200) { sequence_mask_test<float>(8192, 200); }
TEST(sequence_mask_layer, fp16_8192x1000) { sequence_mask_test<__half>(8192, 1000); }
TEST(sequence_mask_layer, fp32_8192x800) { sequence_mask_test<float>(4, 800); }
TEST(sequence_mask_layer, fp16_8192x40) { sequence_mask_test<__half>(8192, 40); }
TEST(sequence_mask_layer, fp32_4096x40) { sequence_mask_test<float>(4096, 40); }
TEST(sequence_mask_layer, fp16_4096x400) { sequence_mask_test<__half>(4096, 400); }
