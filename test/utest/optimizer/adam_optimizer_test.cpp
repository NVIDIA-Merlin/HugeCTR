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

template <typename T>
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


  void update(float* w, const T* g, T* w_tmp) {
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
      float gi = static_cast<float>(g[i]);
      float mi = beta1_ * static_cast<float>(m_[i]) + (1 - beta1_) * gi;
      float vi = beta2_ * static_cast<float>(v_[i]) + (1 - beta2_) * gi * gi;
      m_[i] = mi;
      v_[i] = vi;
      w[i] -= alpha_t * mi / (sqrt(vi) + epsilon_) / scaler;
      if (w_tmp != nullptr)
        w_tmp[i] = w[i];
    }
  }

 private:
  // named as in Algorithm 1 from Adam paper (arXiv:1609.04747)
  vector<T> m_;
  vector<T> v_;
  int len_;
  uint64_t t_;
  const float alpha_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
};

template <typename T>
void compare_array(const T* a, const T* b, int len, float eps) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], eps) << "array differ at index " << i;
  }
}

template <typename T>
void adam_test(int len, int num_update) {
  const int device_id = 0;
  std::shared_ptr<GeneralBuffer<float>> weight_main(new GeneralBuffer<float>(len, device_id));
  std::shared_ptr<GeneralBuffer<T>> weight_sub;
  if (std::is_same<T, float>::value == false) {
    weight_sub.reset(new GeneralBuffer<T>(len, device_id));
  }
  std::shared_ptr<GeneralBuffer<T>> wgrad(new GeneralBuffer<T>(len, device_id));

  std::unique_ptr<float[]> h_weight_main(new float[len]);
  std::unique_ptr<T[]> h_weight_sub;
  if (weight_sub != nullptr) {
    h_weight_sub.reset(new T[len]);
  }
  std::unique_ptr<T[]> h_wgrad(new T[len]);
  std::unique_ptr<float[]> h_weight_expected(new float[len]);
  float* d_weight_main = weight_main->get_ptr_with_offset(0);
  T* d_weight_sub = (weight_sub != nullptr)? weight_sub->get_ptr_with_offset(0) : nullptr;
  T* d_wgrad = wgrad->get_ptr_with_offset(0);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (int i = 0; i < len; ++i) {
    h_weight_expected[i] = h_weight_main[i] = simulator.get_num();
  }
  cudaMemcpy(d_weight_main, h_weight_main.get(), len * sizeof(float), cudaMemcpyHostToDevice);

  AdamOptimizer<T> adam(weight_main, wgrad, weight_sub, device_id);
  AdamCPU<T> adam_cpu(len);
  for (int i = 0; i < num_update; ++i) {
    for (int i = 0; i < len; ++i) {
      h_wgrad[i] = simulator.get_num();
    }
    cudaMemcpy(d_wgrad, h_wgrad.get(), len * sizeof(T), cudaMemcpyHostToDevice);

    adam.update(cudaStreamDefault);
    adam_cpu.update(h_weight_expected.get(), h_wgrad.get(), h_weight_sub.get());
  }

  float eps = 1e-6;
  if (std::is_same<T, __half>::value) {
    eps = 1e-3;
  }

  if (weight_sub != nullptr) {
    std::unique_ptr<T[]> d2h_weight_sub(new T[len]);
    cudaMemcpy(d2h_weight_sub.get(), d_weight_sub, len * sizeof(T), cudaMemcpyDeviceToHost);
    compare_array<T>(d2h_weight_sub.get(), h_weight_sub.get(), len, eps);
  } 

  cudaMemcpy(h_weight_main.get(), d_weight_main, len * sizeof(float), cudaMemcpyDeviceToHost);
  compare_array<float>(h_weight_main.get(), h_weight_expected.get(), len, eps);
}

}  // namespace

TEST(adam, fp32_adam) {
  adam_test<float>(1024, 5);
  adam_test<float>(10240, 5);
}

TEST(adam, fp16_adam) {
  adam_test<__half>(1024, 5);
  adam_test<__half>(10240, 5);
}
