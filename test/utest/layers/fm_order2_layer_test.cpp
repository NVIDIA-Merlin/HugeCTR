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
#include <layers/fm_order2_layer.hpp>
#include <utest/test_utils.hpp>

using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-5f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-1f); }
};

inline float trunc_half(float a) { return __half2float(__float2half(a)); }

void fm_order2_fprop_cpu(const float* in, float* out, int batch_size, int slot_num,
                         int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      float square_sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        float input = in[index];
        sum += input;
        square_sum += input * input;
      }
      float sum_square = sum * sum;
      out[i * emb_vec_size + j] = 0.5f * (sum_square - square_sum);
    }
  }
}

void fm_order2_fprop_cpu(const __half* in, __half* out, int batch_size, int slot_num,
                         int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      float square_sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        float input = __half2float(in[index]);
        sum = trunc_half(sum + input);
        square_sum = trunc_half(square_sum + input * input);
      }
      float sum_square = trunc_half(sum * sum);
      out[i * emb_vec_size + j] = __float2half(0.5f * (sum_square - square_sum));
    }
  }
}

void fm_order2_bprop_cpu(const float* in, const float* top_grad, float* dgrad, int batch_size,
                         int slot_num, int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        sum += in[index];
      }
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        dgrad[index] = top_grad[i * emb_vec_size + j] * (sum - in[index]);
      }
    }
  }
}

void fm_order2_bprop_cpu(const __half* in, const __half* top_grad, __half* dgrad, int batch_size,
                         int slot_num, int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        sum = trunc_half(sum + __half2float(in[index]));
      }
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        dgrad[index] =
            __float2half(__half2float(top_grad[i * emb_vec_size + j]) * (sum - in[index]));
      }
    }
  }
}

template <typename T>
void fm_order2_test(int64_t batch_size, int64_t slot_num, int64_t emb_vec_size) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::Shape in_shape = {batch_size, slot_num * emb_vec_size};
  core23::Tensor bottom_tensor(tensor_params.shape(in_shape));

  core23::Shape out_shape = {batch_size, emb_vec_size};
  core23::Tensor top_tensor(tensor_params.shape(out_shape));

  FmOrder2Layer<T> fm_order2_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  fm_order2_layer.initialize();

  const auto in_len = batch_size * slot_num * emb_vec_size;
  const auto out_len = batch_size * emb_vec_size;
  std::vector<T> h_bottom(in_len);
  std::vector<T> h_top(out_len);
  std::vector<T> h_expected(out_len);
  std::vector<T> h_expected_dgrad(in_len);

  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fm_order2_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  fm_order2_fprop_cpu(h_bottom.data(), h_expected.data(), batch_size, slot_num, emb_vec_size);
  ASSERT_TRUE(
      test::compare_array_approx<T>(h_top.data(), h_expected.data(), out_len, Eps<T>::value()));

  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);
  core23::copy_sync(h_expected_dgrad.data(), h_bottom.data(), h_expected_dgrad.size() * sizeof(T),
                    core23::DeviceType::CPU, core23::DeviceType::CPU);
  test::normal_sync_cpu(h_top.data(), h_top.size(), 0.f, 1.f, generator);

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);
  core23::copy_sync(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                    core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  fm_order2_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_bottom.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());

  fm_order2_bprop_cpu(h_expected_dgrad.data(), h_top.data(), h_expected_dgrad.data(), batch_size,
                      slot_num, emb_vec_size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_bottom.data(), h_expected_dgrad.data(), in_len,
                                            Eps<T>::value()));
}

}  // end of namespace

TEST(fm_order2_layer, fp32_4x2x32) { fm_order2_test<float>(4, 2, 32); }
TEST(fm_order2_layer, fp32_4096x10x64) { fm_order2_test<float>(4096, 10, 64); }
TEST(fm_order2_layer, fp16_4x2x32) { fm_order2_test<__half>(4, 2, 32); }
TEST(fm_order2_layer, fp16_4096x10x64) { fm_order2_test<__half>(4096, 10, 64); }
