/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/relu_layer_half.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-6;

void relu_cpu(const __half* in, __half* out, int len) {
  for (int i = 0; i < len; ++i) {
    if (in[i] < 0) {
      out[i] = 0.0f;
    } else {
      out[i] = in[i];
    }
  }
}

void relu_bprop_cpu(const __half* d_out, __half* d_in, int len) {
  for (int i = 0; i < len; ++i) {
    if (d_in[i] < 0) {
      d_in[i] = 0.0f;
    } else {
      d_in[i] = d_out[i];
    }
  }
}

void relu_test(size_t dim0, size_t dim1) {
  std::shared_ptr<GeneralBuffer<__half>> buf(new GeneralBuffer<__half>());
  vector<size_t> dims = {dim0, dim1};
  std::shared_ptr<Tensor<__half>> in_tensor(new Tensor<__half>(dims, buf));
  std::shared_ptr<Tensor<__half>> out_tensor(new Tensor<__half>(dims, buf));
  buf->init(0);

  const int len = dim0 * dim1;
  __half* d_in = in_tensor->get_ptr();
  __half* d_out = out_tensor->get_ptr();
  std::unique_ptr<__half[]> h_in(new __half[len]);
  std::unique_ptr<__half[]> h_out(new __half[len]);
  std::unique_ptr<__half[]> h_expected(new __half[len]);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  ReluLayerHalf relu_layer(in_tensor, out_tensor, 0);

  // fprop
  for (int i = 0; i < len; ++i) {
    h_in[i] = simulator.get_num();
  }
  cudaMemcpy(d_in, h_in.get(), len * sizeof(__half), cudaMemcpyHostToDevice);
  relu_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, len * sizeof(__half), cudaMemcpyDeviceToHost);

  relu_cpu(h_in.get(), h_expected.get(), len);
  ASSERT_TRUE(test::compare_array_approx<__half>(h_out.get(), h_expected.get(), len, eps));

  // bprop
  for (int i = 0; i < len; ++i) {
    h_in[i] = simulator.get_num();
    h_out[i] = simulator.get_num();
    h_expected[i] = h_in[i];
  }
  cudaMemcpy(d_in, h_in.get(), len * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.get(), len * sizeof(__half), cudaMemcpyHostToDevice);
  relu_layer.bprop(cudaStreamDefault);
  cudaMemcpy(h_in.get(), d_in, len * sizeof(__half), cudaMemcpyDeviceToHost);

  relu_bprop_cpu(h_out.get(), h_expected.get(), len);
  ASSERT_TRUE(test::compare_array_approx<__half>(h_in.get(), h_expected.get(), len, eps));
}

}  // namespace

TEST(relu_layer_half, fprop_and_bprop) {
  relu_test(10, 20);
  relu_test(10, 500);
  relu_test(512, 1024 * 2);
}
