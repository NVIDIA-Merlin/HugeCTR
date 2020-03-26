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

#include "HugeCTR/include/layers/dropout_layer.hpp"

#include <curand.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-6;

void dropout_test(int dim0, int dim1, float rate) {
  curandGenerator_t curand_generator;
  CK_CURAND_THROW_(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));

  std::shared_ptr<GeneralBuffer<float>> buf(new GeneralBuffer<float>());
  vector<int> dims = {dim0, dim1};
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(dims, buf));
  std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(dims, buf));
  buf->init(0);

  const int len = dim0 * dim1;
  const int n_bytes = len * sizeof(float);
  float* d_in = in_tensor->get_ptr();
  float* d_out = out_tensor->get_ptr();

  std::unique_ptr<float[]> h_in(new float[len]);
  std::unique_ptr<float[]> h_out(new float[len]);
  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (int i = 0; i < len; ++i) {
    h_in[i] = simulator.get_num();
  }
  cudaMemcpy(d_in, h_in.get(), n_bytes, cudaMemcpyHostToDevice);

  DropoutLayer dropout_layer(in_tensor, out_tensor, rate, curand_generator, 0);

  std::unique_ptr<float[]> h_mask(new float[len]);
  std::unique_ptr<float[]> h_ref(new float[len]);

  float scale = 1.0 / (1.0 - rate);

  // fprop test
  dropout_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_mask.get(), dropout_layer.mask(), n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out.get(), d_out, n_bytes, cudaMemcpyDeviceToHost);
  int cnt_zero_fprop = 0;
  for (int i = 0; i < len; i++) {
    h_ref[i] = ((1.f - h_mask[i]) >= rate) * h_in[i] * scale;
    if (std::abs(h_ref[i] - 0.f) < 1e-6) {
      cnt_zero_fprop++;
    }
  }
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_ref.get(), len, eps));

  // bprop test
  dropout_layer.bprop(cudaStreamDefault);
  cudaMemcpy(h_in.get(), d_in, n_bytes, cudaMemcpyDeviceToHost);
  int cnt_zero_bprop = 0;
  for (int i = 0; i < len; i++) {
    h_ref[i] = ((1.f - h_mask[i]) >= rate) * h_out[i] * scale;
    if (std::abs(h_ref[i] - 0.f) < 1e-6) {
      cnt_zero_bprop++;
    }
  }
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_ref.get(), len, eps));

  ASSERT_TRUE(cnt_zero_fprop == cnt_zero_bprop);

  CK_CURAND_THROW_(curandDestroyGenerator(curand_generator));
}

TEST(dropout_layer, 32x320_25) { dropout_test(32, 320, 0.25); }

TEST(dropout_layer, 32x320_50) { dropout_test(32, 320, 0.50); }

TEST(dropout_layer, 32x320_75) { dropout_test(32, 320, 0.75); }

TEST(dropout_layer, 32x320_99) { dropout_test(32, 320, 0.99); }

}  // end namespace
