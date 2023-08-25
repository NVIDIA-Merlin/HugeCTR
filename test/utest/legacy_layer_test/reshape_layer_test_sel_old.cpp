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

#include <layers/reshape_layer.hpp>
#include <memory>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void reshape_layer_test(size_t batch_size, size_t n_slot, size_t vector_length,
                        std::vector<int> selected) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  int n_active_slot = selected.empty() ? n_slot : int(selected.size());
  std::vector<size_t> in_dims = {batch_size, n_slot, vector_length};
  std::vector<size_t> out_dims = {batch_size, n_active_slot * vector_length};

  Tensor2<T> in_tensor;
  buff->reserve(in_dims, &in_tensor);
  Tensor2<T> out_tensor;
  ReshapeLayer<T> reshape_layer(in_tensor, out_tensor, buff, selected, test::get_default_gpu());

  buff->allocate();
  reshape_layer.initialize();

  test::GaussianDataSimulator data_sim(0.0f, 1.0f);

  std::vector<T> h_in;
  h_in.resize(in_tensor.get_num_elements());

  data_sim.fill(h_in.data(), h_in.size());

  // fprop
  std::vector<T> h_ref;
  h_ref.resize(batch_size * n_active_slot * vector_length);
  if (selected.empty()) {
    h_ref = h_in;
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      for (int j = 0; j < n_active_slot; j++) {
        for (size_t k = 0; k < vector_length; k++) {
          int in_idx = i * (n_slot * vector_length) + selected[j] * vector_length + k;
          int out_idx = i * (n_active_slot * vector_length) + j * vector_length + k;
          h_ref[out_idx] = h_in[in_idx];
        }
      }
    }
  }

  T* d_in = in_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(d_in, &h_in.front(), in_tensor.get_size_in_bytes(), cudaMemcpyHostToDevice));

  reshape_layer.fprop(true);

  std::vector<T> h_result;
  h_result.resize(batch_size * n_active_slot * vector_length);
  T* d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(&h_result.front(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(
      test::compare_array_approx<T>(&h_result.front(), &h_ref.front(), h_result.size(), eps));

  // bprop
  h_ref.resize(batch_size * n_slot * vector_length);
  h_ref = h_in;

  reshape_layer.bprop();

  h_result.resize(batch_size * n_slot * vector_length);
  cudaMemcpy(&h_result.front(), d_in, in_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(
      test::compare_array_approx<T>(&h_result.front(), &h_in.front(), h_result.size(), eps));
}

}  // namespace

TEST(reshape_layer_old, fp32_selective) {
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

TEST(reshape_layer_old, fp16_selective) {
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
