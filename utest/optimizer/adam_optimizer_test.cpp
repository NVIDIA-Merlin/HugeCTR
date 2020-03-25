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

#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include <vector>
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
using namespace std;
using namespace HugeCTR;

namespace {

class AdamCPU {
 public:
  AdamCPU(int len, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999,
          float epsilon = 1e-8)
      : m_(len),
        v_(len),
        len_(len),
        t_(0),
        alpha_(alpha),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {}

  void update(float* w, const float* g) {
    ++t_;
    const float alpha_t = alpha_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

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
      float gi = g[i];
      float mi = beta1_ * m_[i] + (1 - beta1_) * gi;
      float vi = beta2_ * v_[i] + (1 - beta2_) * gi * gi;
      m_[i] = mi;
      v_[i] = vi;
      w[i] -= (double)alpha_t * mi / (sqrt(vi) + epsilon_) / scaler;
    }
  }

 private:
  // named as in Algorithm 1 from Adam paper (arXiv:1609.04747)
  vector<float> m_;
  vector<float> v_;
  int len_;
  uint64_t t_;
  const float alpha_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

void compare_array(const float* a, const float* b, int len) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], 1e-6) << "array differ at index " << i;
  }
}

void adam_test(int len, int num_update) {
  const int device_id = 0;
  std::shared_ptr<GeneralBuffer<float>> weight(new GeneralBuffer<float>(len, device_id));
  std::shared_ptr<GeneralBuffer<float>> wgrad(new GeneralBuffer<float>(len, device_id));

  std::unique_ptr<float[]> h_weight(new float[len]);
  std::unique_ptr<float[]> h_wgrad(new float[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);
  float* d_weight = weight->get_ptr_with_offset(0);
  float* d_wgrad = wgrad->get_ptr_with_offset(0);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (int i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i] = simulator.get_num();
  }
  cudaMemcpy(d_weight, h_weight.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  AdamOptimizer adam(weight, wgrad, device_id);
  AdamCPU adam_cpu(len);
  for (int i = 0; i < num_update; ++i) {
    for (int i = 0; i < len; ++i) {
      h_wgrad[i] = simulator.get_num();
    }
    cudaMemcpy(d_wgrad, h_wgrad.get(), len * sizeof(float), cudaMemcpyHostToDevice);

    adam.update(cudaStreamDefault);
    adam_cpu.update(h_weight_expected.get(), h_wgrad.get());
  }

  cudaMemcpy(h_weight.get(), d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);
  compare_array(h_weight.get(), h_weight_expected.get(), len);
}

}  // namespace

TEST(adam, adam) {
  adam_test(1024, 5);
  adam_test(10240, 5);
}
