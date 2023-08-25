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

#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/reshape_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void reshape_layer_test(int64_t batch_size, int64_t n_slot, int64_t vector_length,
                        std::vector<int> selected) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  int n_active_slot = selected.empty() ? n_slot : static_cast<int64_t>(selected.size());
  core23::Shape in_shape = {batch_size, n_slot, vector_length};
  core23::Shape out_shape = {batch_size, n_active_slot * vector_length};

  auto device = core23::Device::current();
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::CUDAStream stream(cudaStreamDefault, 0);

  core23::Tensor bottom_tensor(tensor_params.shape(in_shape));
  core23::Tensor top_tensor;
  ReshapeLayer<T> reshape_layer(bottom_tensor, top_tensor, selected, test::get_default_gpu());

  reshape_layer.initialize();

  test::GaussianDataSimulator data_sim(0.0f, 1.0f);

  std::vector<T> h_bottom(bottom_tensor.num_elements());
  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  // fprop
  std::vector<T> h_ref(batch_size * n_active_slot * vector_length);
  if (selected.empty()) {
    h_ref = h_bottom;
  } else {
    for (int64_t i = 0; i < batch_size; i++) {
      for (int j = 0; j < n_active_slot; j++) {
        for (int64_t k = 0; k < vector_length; k++) {
          int in_idx = i * (n_slot * vector_length) + selected[j] * vector_length + k;
          int out_idx = i * (n_active_slot * vector_length) + j * vector_length + k;
          h_ref[out_idx] = h_bottom[in_idx];
        }
      }
    }
  }

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);

  reshape_layer.fprop(true);

  std::vector<T> h_result(batch_size * n_active_slot * vector_length);
  core23::copy_sync(h_result.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  ASSERT_TRUE(
      test::compare_array_approx<T>(&h_result.front(), &h_ref.front(), h_result.size(), eps));

  // bprop
  h_ref.resize(batch_size * n_slot * vector_length);
  h_ref = h_bottom;

  reshape_layer.bprop();

  h_result.resize(batch_size * n_slot * vector_length);
  core23::copy_sync(h_result.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());

  ASSERT_TRUE(
      test::compare_array_approx<T>(&h_result.front(), &h_bottom.front(), h_result.size(), eps));
}

}  // namespace

TEST(reshape_layer, fp32_selective) {
  reshape_layer_test<float>(2, 80, 48, {});
  reshape_layer_test<float>(2, 80, 48, {0, 1, 2});
  reshape_layer_test<float>(2, 80, 48, {0, 1, 3});
  reshape_layer_test<float>(2, 80, 48, {1, 8});
  reshape_layer_test<float>(2, 80, 48, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  reshape_layer_test<float>(2, 81, 48, {});
  reshape_layer_test<float>(2, 81, 48, {0, 1, 2});
  reshape_layer_test<float>(2, 81, 48, {0, 1, 3});
  reshape_layer_test<float>(2, 81, 48, {1, 8});
  reshape_layer_test<float>(2, 81, 48, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  reshape_layer_test<float>(2, 80, 49, {});
  reshape_layer_test<float>(2, 80, 49, {0, 1, 2});
  reshape_layer_test<float>(2, 80, 49, {0, 1, 3});
  reshape_layer_test<float>(2, 80, 49, {1, 8});
  reshape_layer_test<float>(2, 80, 49, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

TEST(reshape_layer, fp16_selective) {
  reshape_layer_test<__half>(2, 80, 48, {});
  reshape_layer_test<__half>(2, 80, 48, {0, 1, 2});
  reshape_layer_test<__half>(2, 80, 48, {0, 1, 3});
  reshape_layer_test<__half>(2, 80, 48, {1, 8});
  reshape_layer_test<__half>(2, 80, 48, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  reshape_layer_test<__half>(2, 81, 48, {});
  reshape_layer_test<__half>(2, 81, 48, {0, 1, 2});
  reshape_layer_test<__half>(2, 81, 48, {0, 1, 3});
  reshape_layer_test<__half>(2, 81, 48, {1, 8});
  reshape_layer_test<__half>(2, 81, 48, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  reshape_layer_test<__half>(2, 80, 49, {});
  reshape_layer_test<__half>(2, 80, 49, {0, 1, 2});
  reshape_layer_test<__half>(2, 80, 49, {0, 1, 3});
  reshape_layer_test<__half>(2, 80, 49, {1, 8});
  reshape_layer_test<__half>(2, 80, 49, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}
