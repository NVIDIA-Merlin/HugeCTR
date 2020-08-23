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

#include "HugeCTR/include/optimizers/nesterov_optimizer.hpp"
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "gtest/gtest.h"
using namespace std;
using namespace HugeCTR;

namespace {

class NesterovCPU {
 public:
  NesterovCPU(int len, float* w, const float* g, const __half* g_half, bool mixed_precision,
              float lr, float mu)
      : w_(w),
        g_(g),
        g_half_(g_half),
        len_(len),
        mixed_precision_(mixed_precision),
        lr_(lr),
        mu_(mu) {
    if (mixed_precision) {
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
        float accum_old = __half2float(accum_half_[i]);
        float accum_new = mu_ * accum_old - lr_ * __half2float(g_half_[i]) / scaler;
        accum_half_[i] = __float2half(accum_new);
        w_[i] += (-mu_ * accum_old + (1 + mu_) * accum_new);
      } else {
        float accum_old = accum_[i];
        float accum_new = mu_ * accum_old - lr_ * g_[i] / scaler;
        accum_[i] = accum_new;
        w_[i] += (-mu_ * accum_old + (1 + mu_) * accum_new);
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

void nesterov_test(size_t len, int num_update, bool mixed_precision) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  Tensor2<float> weight;
  buff->reserve({len}, &weight);
  Tensor2<float> wgrad;
  buff->reserve({len}, &wgrad);
  Tensor2<__half> wgrad_half;
  buff->reserve({len}, &wgrad_half);

  buff->allocate();

  std::unique_ptr<float[]> h_weight(new float[len]);
  std::unique_ptr<float[]> h_wgrad(new float[len]);
  std::unique_ptr<__half[]> h_wgrad_half(new __half[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);
  float* d_weight = weight.get_ptr();
  float* d_wgrad = wgrad.get_ptr();
  __half* d_wgrad_half = wgrad_half.get_ptr();

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (size_t i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight[i] = simulator.get_num();
  }
  cudaMemcpy(d_weight, h_weight.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  NesterovOptimizer nesterov(weight, wgrad, wgrad_half, mixed_precision, 0, 0.01, 0.9);
  NesterovCPU nesterov_cpu(len, h_weight_expected.get(), h_wgrad.get(), h_wgrad_half.get(),
                           mixed_precision, 0.01, 0.9);
  for (int i = 0; i < num_update; ++i) {
    for (size_t i = 0; i < len; ++i) {
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

    nesterov.update(cudaStreamDefault);
    nesterov_cpu.update();
  }

  cudaMemcpy(h_weight.get(), d_weight, len * sizeof(float), cudaMemcpyDeviceToHost);

  if (mixed_precision) {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-3);
  } else {
    compare_array(h_weight.get(), h_weight_expected.get(), len, 1e-6);
  }
}

}  // namespace

TEST(nesterov, fp32_nesterov) {
  nesterov_test(1024, 5, false);
  nesterov_test(10240, 5, false);
}

TEST(nesterov, fp16_nesterov) {
  nesterov_test(1024, 5, true);
  nesterov_test(10240, 5, true);
}
