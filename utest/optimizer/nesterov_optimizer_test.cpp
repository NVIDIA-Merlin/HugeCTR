/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"
#include <vector>
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
using namespace std;
using namespace HugeCTR;

namespace {

class NesterovCPU {
 public:
  NesterovCPU(int len, float lr, float mu) : accum_(len), len_(len), lr_(lr), mu_(mu) {}

  void update(float* w, const float* g) {
    int scaler = 1;
#ifdef SCALE_128
    scaler = 128;
#elif SCALE_256
    scaler = 256;
#elif SCALE_512
    scaler = 512;
#elif SCALE_1024
    scaler = 1024;
#else
    scaler = 1;
#endif

    for (int i = 0; i < len_; ++i) {
      float accum_old = accum_[i];
      accum_[i] = mu_ * accum_old - lr_ * g[i];
      w[i] += (-mu_ * accum_old + (1 + mu_) * accum_[i]) / scaler;
    }
  }

 private:
  vector<float> accum_;
  int len_;
  const float lr_;
  const float mu_;
};

void compare_array(const float* a, const float* b, int len) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], 1e-6) << "array differ at index " << i;
  }
}

void nesterov_test(int len, int num_update) {
  const int device_id = 0;
  GeneralBuffer<float> weight(len, device_id);
  GeneralBuffer<float> wgrad(len, device_id);

  float* h_weight = (float*)malloc(len * sizeof(float));
  float* h_wgrad = (float*)malloc(len * sizeof(float));
  float* h_weight_expected = (float*)malloc(len * sizeof(float));
  float* d_weight = weight.get_ptr_with_offset(0);
  float* d_wgrad = wgrad.get_ptr_with_offset(0);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (int i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i] = simulator.get_num();
  }
  cudaMemcpy(d_weight, h_weight, len * sizeof(float), cudaMemcpyHostToDevice);

  NesterovOptimizer nesterov(weight, wgrad, device_id, 0.01, 0.9);
  NesterovCPU nesterov_cpu(len, 0.01, 0.9);
  for (int i = 0; i < num_update; ++i) {
    for (int i = 0; i < len; ++i) {
      h_wgrad[i] = simulator.get_num();
    }
    cudaMemcpy(d_wgrad, h_wgrad, len * sizeof(float), cudaMemcpyHostToDevice);

    nesterov.update(cudaStreamDefault);
    nesterov_cpu.update(h_weight_expected, h_wgrad);
  }

  cudaMemcpy(h_weight, d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);
  compare_array(h_weight, h_weight_expected, len);

  free(h_weight);
  free(h_wgrad);
  free(h_weight_expected);
}

}  // namespace

TEST(nesterov, nesterov) {
  nesterov_test(1024, 5);
  nesterov_test(10240, 5);
}
