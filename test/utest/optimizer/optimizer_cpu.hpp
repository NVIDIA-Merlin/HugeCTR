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
#pragma once

#include <general_buffer2.hpp>
#include <utils.hpp>

using namespace HugeCTR;

void compare_array(const float* a, const float* b, int len, float eps) {
  for (int i = 0; i < len; ++i) {
    ASSERT_NEAR(a[i], b[i], eps) << "array differ at index " << i;
  }
}

template <typename T>
class AdaGradCPU {
 public:
  AdaGradCPU(int len, float* w, const T* g, const float lr, const float initial_accu_value,
             const float epsilon, const float scaler)
      : w_(w), g_(g), len_(len), lr_(lr), epsilon_(epsilon), scaler_(scaler) {
    sum_.resize(len);
    std::fill(sum_.begin(), sum_.end(), initial_accu_value);
  }

  void update() {
    for (int i = 0; i < len_; ++i) {
      float gi = TypeConvert<float, T>::convert(g_[i]) / scaler_;
      float c_sum = sum_[i];
      c_sum += gi * gi;
      float std_ = epsilon_ + sqrt(c_sum);
      w_[i] -= lr_ * gi / std_;
      sum_[i] = c_sum;
    }
  }

 private:
  // named as in Algorithm 1 from Adam paper (arXiv:1609.04747)
  float* w_;
  const T* g_;
  std::vector<float> sum_;
  const int len_;
  const float lr_;
  const float epsilon_;
  const float scaler_;
};

template <typename T>
class AdamCPU {
 public:
  AdamCPU(int len, float* w, const T* g, float lr = 1.f, float alpha = 0.001, float beta1 = 0.9,
          float beta2 = 0.999, float epsilon = 1e-7, float scaler = 1.f)
      : w_(w),
        g_(g),
        len_(len),
        t_(0),
        lr_(lr),
        alpha_(alpha),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        scaler_(scaler) {
    m_.resize(len);
    v_.resize(len);
  }

  void update() {
    ++t_;
    const float alpha_t = alpha_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

    for (int i = 0; i < len_; ++i) {
      float gi = TypeConvert<float, T>::convert(g_[i]);
      float mi = beta1_ * m_[i] + (1 - beta1_) * gi;
      float vi = beta2_ * v_[i] + (1 - beta2_) * gi * gi;
      m_[i] = mi;
      v_[i] = vi;
      w_[i] -= alpha_t * mi / (sqrt(vi) + epsilon_) / scaler_;
    }
  }

 private:
  // named as in Algorithm 1 from Adam paper (arXiv:1609.04747)
  float* w_;
  const T* g_;
  std::vector<float> m_;
  std::vector<float> v_;
  const int len_;
  uint64_t t_;
  const float lr_;
  const float alpha_;
  const float beta1_;
  const float beta2_;
  const float epsilon_;
  const float scaler_;
};

template <typename T>
class FtrlCPU {
 public:
  FtrlCPU(int len, float* w, const T* g, float lr = 1.f, float alpha = 0.001, float beta = 0.0f,
          float lambda1 = 0.0f, float lambda2 = 0.0f, float scaler = 1.0f)
      : w_(w),
        g_(g),
        len_(len),
        lr_(lr),
        alpha_(alpha),
        beta_(beta),
        lambda1_(lambda1),
        lambda2_(lambda2),
        scaler_(scaler) {
    z_.resize(len);
    n_.resize(len);
  }
  void update() {
    for (int i = 0; i < len_; ++i) {
      float gi = TypeConvert<float, T>::convert(g_[i]) / scaler_;
      float ni_new = n_[i] + gi * gi;
      float zi = z_[i] + gi + (sqrt(n_[i]) - sqrt(ni_new)) * w_[i] / alpha_;
      float x = lambda1_ * (1.0f - 2.0f * signbit(zi)) - zi;
      float y = sqrt(ni_new) / alpha_ + lambda2_ + beta_ / alpha_;
      n_[i] = ni_new;
      z_[i] = zi;
      w_[i] = x / y * signbit(lambda1_ - abs(zi));
    }
  }

 private:
  // named as in ftrl paper Ad Click Prediction: a View from the Trenches
  // except alpha is lr_
  float* w_;
  const T* g_;
  std::vector<float> z_;
  std::vector<float> n_;
  const int len_;

  const float lr_;
  const float alpha_;
  const float beta_;
  const float lambda1_;
  const float lambda2_;
  const float scaler_;
};

template <typename T>
class MomentumSGDCPU {
 public:
  MomentumSGDCPU(int len, float* w, T* g, float lr = 0.01, float mu = 0.9, float scaler = 1.f)
      : w_(w), g_(g), len_(len), lr_(lr), mu_(mu), scaler_(scaler) {
    accum_.resize(len);
  }

  void update() {
    for (int i = 0; i < len_; ++i) {
      float acc = mu_ * accum_[i] - lr_ * TypeConvert<float, T>::convert(g_[i]) / scaler_;
      accum_[i] = acc;
      w_[i] += acc;
    }
  }

 private:
  float* w_;
  const T* g_;
  std::vector<float> accum_;
  const int len_;
  const float lr_;
  const float mu_;
  const float scaler_;
};
template <typename T>
class NesterovCPU {
 public:
  NesterovCPU(int len, float* w, const T* g, float lr, float mu, float scaler)
      : w_(w), g_(g), len_(len), lr_(lr), mu_(mu), scaler_(scaler) {
    accum_.resize(len);
  }

  void update() {
    for (int i = 0; i < len_; ++i) {
      float accum_old = accum_[i];
      float accum_new = mu_ * accum_old - lr_ * TypeConvert<float, T>::convert(g_[i]) / scaler_;
      accum_[i] = accum_new;
      w_[i] += (-mu_ * accum_old + (1 + mu_) * accum_new);
    }
  }

 private:
  float* w_;
  const T* g_;
  std::vector<float> accum_;
  const int len_;
  const float lr_;
  const float mu_;
  const float scaler_;
};
template <typename T>
class SGDCPU {
 public:
  SGDCPU(int len, float* w, const T* g, float lr = 0.001, float scaler = 1.f)
      : w_(w), g_(g), len_(len), lr_(lr), scaler_(scaler) {}

  void update() {
    for (int i = 0; i < len_; ++i) {
      float gi = TypeConvert<float, T>::convert(g_[i]) / scaler_;
      w_[i] -= lr_ * gi;
    }
  }

 private:
  float* w_;
  const T* g_;
  const int len_;
  const float lr_;
  const float scaler_;
};
