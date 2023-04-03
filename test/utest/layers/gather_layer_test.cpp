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

#include <layers/gather_layer.hpp>
#include <memory>
#include <set>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void gather_layer_test(int64_t dimension, int64_t height, int64_t width, std::vector<int> indices) {
  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params = core23::TensorParams()
                                           .device(device)
                                           .data_type(core23::ScalarType::Float)
                                           .buffer_channel(core23::GetRandomBufferChannel());

  const int64_t ts = height * width;
  const size_t n_outs = indices.size();
  core23::Shape in_shape = {dimension, ts};
  core23::Tensor bottom_tensor(tensor_params.shape(in_shape));
  core23::Shape out_shape = {(int64_t)n_outs, ts};
  core23::Tensor top_tensor(tensor_params.shape(out_shape));

  const int64_t output_size = n_outs * ts;
  const int64_t input_size = dimension * ts;
  std::unique_ptr<T[]> h_in(new T[input_size]);
  std::unique_ptr<T[]> h_refs(new T[output_size]);
  std::unique_ptr<T[]> d2h_top(new T[output_size]);
  std::unique_ptr<T[]> h_bottom(new T[input_size]);

  test::normal_sync_cpu(h_in.get(), input_size, 0.f, 1.f, generator);

  GatherLayer<T> gather_layer(bottom_tensor, top_tensor, indices, test::get_default_gpu());

  gather_layer.initialize();

  // fprop
  for (auto j = 0; j < n_outs; j++) {
    for (auto i = 0; i < ts; i++) {
      h_refs.get()[j * ts + i] = h_in.get()[indices.data()[j] * ts + i];
    }
  }
  T* d_in = bottom_tensor.data<T>();
  core23::copy_sync(d_in, h_in.get(), bottom_tensor.num_bytes(), bottom_tensor.device(),
                    core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gather_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T* d_out = top_tensor.data<T>();
  core23::copy_sync(d2h_top.get(), d_out, top_tensor.num_bytes(), core23::DeviceType::CPU,
                    top_tensor.device());

  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_refs.get(), output_size, eps));

  // bprop
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gather_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (unsigned int i = 0; i < input_size; i++) {
    h_in.get()[i] = 0.0f;
  }
  for (auto j = 0; j < n_outs; j++) {
    for (auto i = 0; i < ts; i++) {
      h_in.get()[indices.data()[j] * ts + i] = h_refs.get()[j * ts + i];
    }
  }

  core23::copy_sync(h_bottom.get(), d_in, bottom_tensor.num_bytes(), core23::DeviceType::CPU,
                    bottom_tensor.device());
  ASSERT_TRUE(test::compare_array_approx<T>(h_bottom.get(), h_in.get(), input_size, eps));
}

}  // namespace

// TEST(gather_layer, fp32_20x20x64_2) {
//  std::vector<int> indices{0, 23};
//  gather_layer_test<float>(20, 20, 64, indices);
//}

// TODO failed case
TEST(gather_layer, fp32_5x2x3_2) {
  std::vector<int> indices{0, 2};
  gather_layer_test<float>(5, 2, 3, indices);
}
TEST(gather_layer, fp32_64x64x64_1) {
  std::vector<int> indices{32};
  gather_layer_test<float>(64, 64, 64, indices);
}
TEST(gather_layer, fp32_64x64x64_4) {
  std::vector<int> indices{0, 23, 43, 63};
  gather_layer_test<float>(64, 64, 64, indices);
}
TEST(gather_layer, fp32_64x512x512_7) {
  std::vector<int> indices{0, 15, 17, 22, 23, 32, 63};
  gather_layer_test<float>(64, 512, 512, indices);
}
TEST(gather_layer, fp32_512x1024x1024_11) {
  std::vector<int> indices{0, 15, 17, 22, 23, 32, 63, 88, 123, 400, 511};
  gather_layer_test<float>(512, 1024, 1024, indices);
}
TEST(gather_layer, fp32_2048x1024x1024_14) {
  std::vector<int> indices{0, 15, 17, 22, 23, 32, 63, 88, 123, 400, 511, 1024, 2040, 2047};
  gather_layer_test<float>(2048, 1024, 1024, indices);
}