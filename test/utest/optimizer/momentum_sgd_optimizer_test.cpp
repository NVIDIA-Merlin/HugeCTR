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

#include "HugeCTR/include/optimizers/momentum_sgd_optimizer.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

class SGDCPU {
 public:
  SGDCPU(int len, float* w, float* g, __half* g_half, bool mixed_precision, float lr = 0.01,
         float mu = 0.9)
      : w_(w),
        g_(g),
        g_half_(g_half),
        len_(len),
        mixed_precision_(mixed_precision),
        lr_(lr),
        mu_(mu) {
    if (mixed_precision_) {
      accum_half_.resize(len);
    } else {
      accum_.resize(len);
    }
  }

  void update() {
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
        float acc = mu_ * __half2float(accum_half_[i]) - lr_ * __half2float(g_half_[i]) / scaler;
        accum_half_[i] = __float2half(acc);
        w_[i] += acc;
      } else {
        float acc = mu_ * accum_[i] - lr_ * g_[i] / scaler;
        accum_[i] = acc;
        w_[i] += acc;
      }
    }
  }

 private:
  float* w_;
  const float* g_;
  const __half* g_half_;
  vector<float> accum_;
  vector<__half> accum_half_;
  const int len_;
  const bool mixed_precision_;
  const float lr_;
  const float mu_;
};

void compare_array(const float* a, const float* b, int len, float eps) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], eps) << "array differ at index " << i;
  }
}

void sgd_test(size_t len, int num_update, bool mixed_precision) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  Tensor2<float> weight;
  buff->reserve({len}, &weight);
  Tensor2<float> wgrad;
  buff->reserve({len}, &wgrad);
  Tensor2<__half> wgrad_half;
  buff->reserve({len}, &wgrad_half);

  MomentumSGDOptimizer sgd(weight, wgrad, wgrad_half, mixed_precision, buff,
                           test::get_default_gpu(), 0.01, 0.9);

  buff->allocate();

  sgd.initialize();

  std::unique_ptr<float[]> h_weight(new float[len]);
  std::unique_ptr<float[]> h_wgrad(new float[len]);
  std::unique_ptr<__half[]> h_wgrad_half(new __half[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);
  float* d_weight = weight.get_ptr();
  float* d_wgrad = wgrad.get_ptr();
  __half* d_wgrad_half = wgrad_half.get_ptr();

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_weight.get(), len);
  for (size_t i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i];
  }
  cudaMemcpy(d_weight, h_weight.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  SGDCPU sgd_cpu(len, h_weight_expected.get(), h_wgrad.get(), h_wgrad_half.get(), mixed_precision,
                 0.01, 0.9);
  for (int i = 0; i < num_update; ++i) {
    simulator.fill(h_wgrad.get(), len);
    if (mixed_precision) {
      for (size_t i = 0; i < len; ++i) {
        h_wgrad_half[i] = __float2half(h_wgrad[i]);
      }
      cudaMemcpy(d_wgrad_half, h_wgrad_half.get(), len * sizeof(__half), cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(d_wgrad, h_wgrad.get(), len * sizeof(float), cudaMemcpyHostToDevice);
    }

    sgd.update();
    sgd_cpu.update();
  }

  cudaMemcpy(h_weight.get(), d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);
  if (mixed_precision) {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-3);
  } else {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-6);
  }
}

}  // namespace

TEST(optimizer_test, fp32_momonetum_sgd) {
  sgd_test(1024, 5, false);
  sgd_test(10240, 5, false);
}

TEST(optimizer_test, fp16_momonetum_sgd) {
  sgd_test(1024, 5, true);
  sgd_test(10240, 5, true);
}
