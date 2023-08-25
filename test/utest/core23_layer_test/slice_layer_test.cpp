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
#include <layers/slice_layer.hpp>
#include <memory>
#include <set>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void slice_layer_test(int64_t height, int64_t width, std::vector<std::pair<int, int>> ranges) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::Shape in_shape = {height, width};

  core23::Tensor bottom_tensor(tensor_params.shape(in_shape));

  std::vector<T> h_bottom(bottom_tensor.num_elements(), 0.0);
  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  std::vector<core23::Tensor> top_tensors;
  SliceLayer<T> slice_layer(bottom_tensor, top_tensors, ranges, test::get_default_gpu());

  slice_layer.initialize();

  size_t n_outs = top_tensors.size();

  // fprop
  std::vector<std::vector<T>> h_refs;
  for (size_t i = 0; i < n_outs; i++) {
    std::vector<T> h_ref(top_tensors[i].num_elements(), 0.0);
    h_refs.push_back(h_ref);
  }

  int i = 0;
  for (auto& range : ranges) {
    int out_width = range.second - range.first;
    for (size_t r = 0; r < height; r++) {
      for (int c = range.first; c < range.second; c++) {
        int in_idx = r * width + c;
        int out_idx = r * out_width + c - range.first;
        h_refs[i][out_idx] = h_bottom[in_idx];
      }
    }
    i++;
  }

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);

  slice_layer.fprop(true);

  for (size_t i = 0; i < n_outs; i++) {
    std::vector<T> h_top(top_tensors[i].num_elements(), 0.0);
    core23::copy_sync(h_top.data(), top_tensors[i].data(), top_tensors[i].num_bytes(),
                      core23::DeviceType::CPU, top_tensors[i].device());
    ASSERT_TRUE(
        test::compare_array_approx<T>(&h_top.front(), &h_refs[i].front(), h_top.size(), eps));
  }

  // bprop
  slice_layer.bprop();

  for (unsigned int i = 0; i < h_bottom.size(); i++) {
    h_bottom[i] = 0.0f;
  }
  i = 0;
  for (auto& range : ranges) {
    int out_width = range.second - range.first;
    for (size_t r = 0; r < height; r++) {
      for (int c = range.first; c < range.second; c++) {
        int in_idx = r * width + c;
        int out_idx = r * out_width + c - range.first;
        h_bottom[in_idx] = h_bottom[in_idx] + h_refs[i][out_idx];
      }
    }
    i++;
  }
  std::vector<T> h_top(bottom_tensor.num_elements(), 0.0);
  core23::copy_sync(h_top.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());
  ASSERT_TRUE(test::compare_array_approx<T>(&h_top.front(), &h_bottom.front(), h_top.size(), eps));
}

}  // namespace

TEST(slice_layer, fp32_64x128_0_48_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 48));
  ranges.push_back(std::make_pair(32, 64));
  slice_layer_test<float>(64, 128, ranges);
}

TEST(slice_layer, fp32_64x128_0_32_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 32));
  ranges.push_back(std::make_pair(32, 64));
  slice_layer_test<float>(64, 128, ranges);
}

TEST(slice_layer, fp32_64x100_0_40_50_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 40));
  ranges.push_back(std::make_pair(50, 90));
  slice_layer_test<float>(64, 100, ranges);
}

TEST(slice_layer, fp32_64x100_0_50_40_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 50));
  ranges.push_back(std::make_pair(40, 90));
  slice_layer_test<float>(64, 100, ranges);
}

TEST(slice_layer, fp32_64x256_0_50_40_90_80_130) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 50));
  ranges.push_back(std::make_pair(40, 90));
  ranges.push_back(std::make_pair(80, 130));
  slice_layer_test<float>(64, 256, ranges);
}

TEST(slice_layer, fp32_64x256_0_32_64_80_96_128_128_160_192_256) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 32));
  ranges.push_back(std::make_pair(64, 80));
  ranges.push_back(std::make_pair(96, 128));
  ranges.push_back(std::make_pair(128, 160));
  ranges.push_back(std::make_pair(192, 256));
  slice_layer_test<float>(64, 256, ranges);
}

TEST(slice_layer, fp16_64x128_0_48_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 48));
  ranges.push_back(std::make_pair(32, 64));
  slice_layer_test<__half>(64, 128, ranges);
}

TEST(slice_layer, fp16_64x128_0_32_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 32));
  ranges.push_back(std::make_pair(32, 64));
  slice_layer_test<__half>(64, 128, ranges);
}

TEST(slice_layer, fp16_64x100_0_40_50_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 40));
  ranges.push_back(std::make_pair(50, 90));
  slice_layer_test<__half>(64, 100, ranges);
}

TEST(slice_layer, fp16_64x100_0_50_40_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 50));
  ranges.push_back(std::make_pair(40, 90));
  slice_layer_test<__half>(64, 100, ranges);
}

TEST(slice_layer, fp16_64x256_0_50_40_90_80_130) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 50));
  ranges.push_back(std::make_pair(40, 90));
  ranges.push_back(std::make_pair(80, 130));
  slice_layer_test<__half>(64, 256, ranges);
}

TEST(slice_layer, fp16_64x256_0_32_64_80_96_128_128_160_192_256) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0, 32));
  ranges.push_back(std::make_pair(64, 80));
  ranges.push_back(std::make_pair(96, 128));
  ranges.push_back(std::make_pair(128, 160));
  ranges.push_back(std::make_pair(192, 256));
  slice_layer_test<__half>(64, 256, ranges);
}
