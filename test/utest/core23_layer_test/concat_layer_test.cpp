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

#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/concat_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void concat_layer_test(int64_t height, std::vector<int64_t> widths) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  std::vector<core23::Tensor> bottom_tensors;

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<std::vector<T>> h_bottoms;

  int64_t n_ins = widths.size();

  int64_t new_width = 0;
  for (int64_t i = 0; i < n_ins; i++) {
    int64_t width = widths[i];
    new_width += width;
    core23::Shape in_shape = {height, width};
    bottom_tensors.emplace_back(tensor_params.shape(in_shape));

    std::vector<T> h_bottom(in_shape.size(), static_cast<T>(0.f));
    test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);
    h_bottoms.push_back(h_bottom);
  }
  core23::Tensor top_tensor;
  ConcatLayer<T> concat_layer(bottom_tensors, top_tensor, test::get_default_gpu());

  concat_layer.initialize();

  // fprop
  std::vector<T> h_ref(top_tensor.num_elements(), 0.0);
  for (int64_t r = 0; r < height; r++) {
    for (int64_t c = 0; c < new_width; c++) {
      int64_t out_idx = r * new_width + c;
      int64_t in_no = 0;
      int64_t c2 = c;
      int64_t accum_width = 0;
      for (int64_t k = 0; k < n_ins; k++) {
        if (c < accum_width + widths[k]) {
          in_no = k;
          c2 -= accum_width;
          break;
        }
        accum_width += widths[k];
      }
      int in_idx = r * widths[in_no] + c2;
      h_ref[out_idx] = h_bottoms[in_no][in_idx];
    }
  }

  for (int64_t i = 0; i < n_ins; i++) {
    core23::copy_sync(bottom_tensors[i].data(), h_bottoms[i].data(), bottom_tensors[i].num_bytes(),
                      bottom_tensors[i].device(), core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  concat_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<T> h_top(top_tensor.num_elements(), 0.0);
  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_ref.data(), h_top.size(), eps));

  // bprop
  concat_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  for (int64_t i = 0; i < n_ins; i++) {
    core23::fill_sync(top_tensor.data<T>(), top_tensor.num_elements(),
                      core23::TypeConverter<T, float>::value(0.f), top_tensor.device());
  }
  concat_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_ref.data(), h_top.size(), eps));
}

}  // namespace

TEST(concat_layer, fp32_64x32_64x32) { concat_layer_test<float>(64, {32, 32}); }

TEST(concat_layer, fp32_5x32_5x32) { concat_layer_test<float>(5, {32, 32}); }

TEST(concat_layer, fp32_4096x640_4096x1280) { concat_layer_test<float>(4096, {640, 1280}); }

TEST(concat_layer, fp32_64x32_64x64_64x96) { concat_layer_test<float>(64, {32, 64, 96}); }

TEST(concat_layer, fp32_64x32_64x64_64x32_64x128) {
  concat_layer_test<float>(64, {32, 64, 32, 128});
}

TEST(concat_layer, fp32_64x32_64x64_64x32_64x128_64x256) {
  concat_layer_test<float>(64, {32, 64, 32, 128, 256});
}

TEST(concat_layer, fp16_64x32_64x32) { concat_layer_test<__half>(64, {32, 32}); }

TEST(concat_layer, fp16_5x32_5x32) { concat_layer_test<__half>(5, {32, 32}); }

TEST(concat_layer, fp16_4096x640_4096x1280) { concat_layer_test<__half>(4096, {640, 1280}); }

TEST(concat_layer, fp16_64x32_64x64_64x96) { concat_layer_test<__half>(64, {32, 64, 96}); }

TEST(concat_layer, fp16_64x32_64x64_64x32_64x128) {
  concat_layer_test<__half>(64, {32, 64, 32, 128});
}

TEST(concat_layer, fp16_64x32_64x64_64x32_64x128_64x256) {
  concat_layer_test<__half>(64, {32, 64, 32, 128, 256});
}
