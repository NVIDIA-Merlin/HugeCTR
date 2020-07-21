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
  AdamCPU(int len, float* w, const float* g, const __half* g_half, bool mixed_precision,
          float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
      : w_(w),
        g_(g),
        g_half_(g_half),
        len_(len),
        mixed_precision_(mixed_precision),
        t_(0),
        alpha_(alpha),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {
    if (mixed_precision) {
      m_half_.resize(len);
      v_half_.resize(len);
    } else {
      m_.resize(len);
      v_.resize(len);
    }
  }

  void update() {
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
      if (mixed_precision_) {
        float gi = __half2float(g_half_[i]);
        float mi = beta1_ * __half2float(m_half_[i]) + (1 - beta1_) * gi;
        float vi = beta2_ * __half2float(v_half_[i]) + (1 - beta2_) * gi * gi;
        m_half_[i] = __float2half(mi);
        v_half_[i] = __float2half(vi);
        w_[i] -= alpha_t * mi / (sqrt(vi) + epsilon_) / scaler;
      } else {
        float gi = g_[i];
        float mi = beta1_ * m_[i] + (1 - beta1_) * gi;
        float vi = beta2_ * v_[i] + (1 - beta2_) * gi * gi;
        m_[i] = mi;
        v_[i] = vi;
        w_[i] -= alpha_t * mi / (sqrt(vi) + epsilon_) / scaler;
      }
    }
  }

 private:
  // named as in Algorithm 1 from Adam paper (arXiv:1609.04747)
  float* w_;
  const float* g_;
  const __half* g_half_;
  vector<float> m_;
  vector<float> v_;
  vector<__half> m_half_;
  vector<__half> v_half_;
  const int len_;
  const bool mixed_precision_;
  uint64_t t_;
  const float alpha_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

void compare_array(const float* a, const float* b, int len, float eps) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], eps) << "array differ at index " << i;
  }
}

void adam_test(int len, int num_update, bool mixed_precision) {
  const int device_id = 0;
  std::shared_ptr<GeneralBuffer<float>> weight(new GeneralBuffer<float>(len, device_id));
  std::shared_ptr<GeneralBuffer<float>> wgrad(new GeneralBuffer<float>(len, device_id));
  std::shared_ptr<GeneralBuffer<__half>> wgrad_half(new GeneralBuffer<__half>(len, device_id));

  std::unique_ptr<float[]> h_weight(new float[len]);
  std::unique_ptr<float[]> h_wgrad(new float[len]);
  std::unique_ptr<__half[]> h_wgrad_half(new __half[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);
  float* d_weight = weight->get_ptr_with_offset(0);
  float* d_wgrad = wgrad->get_ptr_with_offset(0);
  __half* d_wgrad_half = wgrad_half->get_ptr_with_offset(0);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (int i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i] = simulator.get_num();
  }
  cudaMemcpy(d_weight, h_weight.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  AdamOptimizer adam(weight, wgrad, wgrad_half, mixed_precision, device_id);
  AdamCPU adam_cpu(len, h_weight_expected.get(), h_wgrad.get(), h_wgrad_half.get(),
                   mixed_precision);
  for (int i = 0; i < num_update; ++i) {
    for (int i = 0; i < len; ++i) {
      float val = simulator.get_num();
      if (mixed_precision) {
        h_wgrad_half[i] = __float2half(val);
      } else {
        h_wgrad[i] = val;
      }
    }

    if (mixed_precision) {
      cudaMemcpy(d_wgrad_half, h_wgrad_half.get(), len * sizeof(__half), cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(d_wgrad, h_wgrad.get(), len * sizeof(float), cudaMemcpyHostToDevice);
    }

    adam.update(cudaStreamDefault);
    adam_cpu.update();
  }

  cudaMemcpy(h_weight.get(), d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);
  if (mixed_precision) {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-3);
  } else {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-6);
  }
}

}  // namespace

TEST(adam, fp32_adam) {
  adam_test(1024, 5, false);
  adam_test(10240, 5, false);
}

TEST(adam, fp16_adam) {
  adam_test(1024, 5, true);
  adam_test(10240, 5, true);
}
