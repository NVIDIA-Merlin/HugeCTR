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
    input[i] = abs(floor(__half2float(input[i]) * max_sequence_len));
  }
}

template <typename T>
void sequence_mask_cpu(T* seq_len_from, T* seq_len_to, T* output, size_t batch_size,
                       size_t max_sequence_len_from, size_t max_sequence_len_to, size_t out_size) {
  for (size_t i = 0; i < batch_size; i++) {
    float length_from = seq_len_from[i];
    float length_to = seq_len_to[i];
    for (size_t j = 0; j < max_sequence_len_from; j++) {
      for (size_t k = 0; k < max_sequence_len_to; k++) {
        if (j < length_from && k < length_to) {
          output[i * max_sequence_len_from * max_sequence_len_to + j * max_sequence_len_to + k] =
              (T)(1.0f);
        } else {
          output[i * max_sequence_len_from * max_sequence_len_to + j * max_sequence_len_to + k] =
              (T)(0.0f);
        }
      }
    }
  }
}

template <typename T>
void sequence_mask_test(int64_t batch_size, int max_sequence_len_from, int max_sequence_len_to) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::Shape shape_in = {batch_size};
  core23::Shape shape_out = {batch_size, 1, max_sequence_len_from, max_sequence_len_to};
  auto in_size = shape_in.size();
  auto out_size = shape_out.size();

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  // core23::Tensor bottom_tensor(tensor_params.shape(shape_in));
  std::vector<core23::Tensor> bottom_tensors;
  bottom_tensors.emplace_back(tensor_params.shape(shape_in));
  bottom_tensors.emplace_back(tensor_params.shape(shape_in));
  core23::Tensor top_tensor(tensor_params.shape(shape_out));

  SequenceMaskLayer<T> sequence_mask_layer(bottom_tensors, top_tensor, max_sequence_len_from,
                                           max_sequence_len_to, test::get_default_gpu());

  auto* d_seq_len_from = bottom_tensors[0].data<T>();
  auto* d_seq_len_to = bottom_tensors[1].data<T>();
  auto* d_top = top_tensor.data<T>();

  std::vector<T> h_seq_len_from(in_size);
  std::vector<T> h_seq_len_to(in_size);
  std::vector<T> h_top(out_size);
  std::vector<T> h_ref(out_size);

  // fprop
  test::normal_sync_cpu(h_seq_len_from.data(), in_size, 0.f, 1.f, generator);
  f2i_input(h_seq_len_from.data(), in_size, max_sequence_len_from);
  core23::copy_sync(d_seq_len_from, h_seq_len_from.data(), bottom_tensors[0].num_bytes(),
                    bottom_tensors[0].device(), core23::DeviceType::CPU);

  test::normal_sync_cpu(h_seq_len_to.data(), in_size, 0.f, 1.f, generator);
  f2i_input(h_seq_len_to.data(), in_size, max_sequence_len_to);
  core23::copy_sync(d_seq_len_to, h_seq_len_to.data(), bottom_tensors[1].num_bytes(),
                    bottom_tensors[1].device(), core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  sequence_mask_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), d_top, top_tensor.num_bytes(), core23::DeviceType::CPU,
                    top_tensor.device());

  sequence_mask_cpu(h_seq_len_from.data(), h_seq_len_to.data(), h_ref.data(), batch_size,
                    max_sequence_len_from, max_sequence_len_to, out_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_ref.data(), out_size, Eps<T>::value()));
}

}  // namespace

TEST(sequence_mask_layer, fp32_8192x200) { sequence_mask_test<float>(8192, 200, 200); }
TEST(sequence_mask_layer, fp16_8192x1000) { sequence_mask_test<__half>(8192, 1000, 800); }
TEST(sequence_mask_layer, fp32_8192x800) { sequence_mask_test<float>(4, 800, 800); }
TEST(sequence_mask_layer, fp16_8192x40) { sequence_mask_test<__half>(8192, 40, 100); }
TEST(sequence_mask_layer, fp32_4096x40) { sequence_mask_test<float>(4096, 40, 40); }
TEST(sequence_mask_layer, fp16_4096x400) { sequence_mask_test<__half>(4096, 400, 200); }
