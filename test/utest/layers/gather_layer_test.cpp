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
void gather_layer_test(size_t dimention, size_t height, size_t width, std::vector<int> indices) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  const size_t ts = height * width;
  const size_t n_outs = indices.size();
  std::vector<size_t> in_dims = {dimention, ts};
  Tensor2<T> in_tensor;
  buff->reserve(in_dims, &in_tensor);
  Tensor2<T> out_tensor;
  std::vector<size_t> out_dims = {n_outs, ts};
  buff->reserve(out_dims, &out_tensor);

  const size_t output_size = n_outs * ts;
  const size_t input_size = dimention * ts;
  std::unique_ptr<T[]> h_in(new T[input_size]);
  std::unique_ptr<T[]> h_refs(new T[output_size]);
  std::unique_ptr<T[]> d2h_top(new T[output_size]);
  std::unique_ptr<T[]> h_bottom(new T[input_size]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_in.get(), input_size);

  GatherLayer<T> gather_layer(in_tensor, out_tensor, buff, indices, test::get_default_gpu());

  buff->allocate();
  gather_layer.initialize();

  // fprop
  for (size_t j = 0; j < n_outs; j++) {
    for (size_t i = 0; i < ts; i++) {
      h_refs.get()[j * ts + i] = h_in.get()[indices.data()[j] * ts + i];
    }
  }
  T* d_in = in_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(d_in, h_in.get(), in_tensor.get_size_in_bytes(), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gather_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  T* d_out = out_tensor.get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_top.get(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_refs.get(), output_size, eps));

  // bprop
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  gather_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (unsigned int i = 0; i < input_size; i++) {
    h_in.get()[i] = 0.0f;
  }
  for (size_t j = 0; j < n_outs; j++) {
    for (size_t i = 0; i < ts; i++) {
      h_in.get()[indices.data()[j] * ts + i] = h_refs.get()[j * ts + i];
    }
  }

  HCTR_LIB_THROW(
      cudaMemcpy(h_bottom.get(), d_in, in_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));
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