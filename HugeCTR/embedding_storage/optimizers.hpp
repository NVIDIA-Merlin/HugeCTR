/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cmath>
#include <cstdint>

namespace embedding {

/**
 * SGD (Stateless)
 * ---------------
 * g_i = -eta * g_i / s
 */
inline void sgd_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr, float scaler,
                            float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;

    g[i] = -lr * gi;
  }
}

/**
 * Momentum SGD
 * ------------
 * v_i = beta * v_i + g_i / s
 * g_i = -eta * v_i
 */
inline void momentum_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr,
                                 float momentum_decay, float** state_tensors, float scaler,
                                 float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* m = state_tensors[idx] - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float mi = m[i] = momentum_decay * m[i] - lr * gi;

    g[i] = mi;
  }
}

/**
 * Nesterov Momentum
 * -----------------
 * w*_i = w_i + beta * v_i
 * g = f(w*)
 * v_i = beta * v_i + g_i / s
 * g_i = -eta * v_i
 */
inline void nesterov_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr,
                                 float momentum_decay, float** state_tensors, float scaler,
                                 float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* m = state_tensors[idx] - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float mi_prev = m[i];
    float mi = m[i] = momentum_decay * mi_prev - lr * gi;

    g[i] = mi + momentum_decay * mi - momentum_decay * mi_prev;
  }
}

/**
 * AdaGrad
 * -------
 * g_i = g_i / s
 * v_i = v_i + g_i^2
 * g_i = -eta * g_i / (sqrt(v_i) + epsilon)
 */
inline void ada_grad_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr,
                                 float** state_tensors, float epsilon, float scaler, float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* v = state_tensors[idx] - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float vi = v[i] = v[i] + gi * gi;

    g[i] = -lr * gi / (std::sqrt(vi) + epsilon);
  }
}

/**
 * RMSProp
 * -------
 * g_i = g_i / s
 * v_i = beta * v_i + (1 - beta) * g_i^2
 * g_i = -eta * g_i / (sqrt(v_i) + epsilon)
 */
inline void rms_prop_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr, float beta,
                                 float** state_tensors, float epsilon, float scaler, float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* v = state_tensors[idx] - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float vi = v[i] = beta * v[i] + (1.f - beta) * gi * gi;

    g[i] = -lr * gi / (std::sqrt(vi) + epsilon);
  }
}

/**
 * Adam
 * ----
 * g_i = g_i / s
 * m_i = beta_1 * m_i + (1 - beta_1) * g_i
 * v_i = beta_2 * v_i + (1 - beta_2) * g_i^2
 *
 * m_i_debiased = m_i / (1 - beta_1^t)
 * v_i_debiased = v_i / (1 - beta_2^t)
 *
 * g_i = -eta * m_i_debiased / (sqrt(v_i_debiased) + epsilon)
 */
inline void adam_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr_scaled_bias,
                             float beta1, float beta2, float** state_tensors, float epsilon,
                             float scaler, float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* m = state_tensors[idx] - start;
  float* v = m + end - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float mi = m[i] = beta1 * m[i] + (1.f - beta1) * gi;
    float vi = v[i] = beta2 * v[i] + (1.f - beta2) * gi * gi;

    g[i] = -lr_scaled_bias * mi / (std::sqrt(vi) + epsilon);
  }
}

/**
 * FTRL
 * ----
 * g_i = g_i / s
 * sigma = (sqrt(n_i + g_i^2) - sqrt(n_i)) / eta
 * n_i = n_i + g_i^2
 * z_i = z_i + g_i - sigma w_i
 *
 * if abs(z_i) <= lambda_1:
 *   w_i = 0
 * else:
 *   w_i = -sqrt((beta + sqrt(n_i)) / eta + lambda_2) * (z_i - sign(z_i) * lambda_1)
 */
inline void ftrl_update_grad(uint32_t idx, const uint32_t* ev_offsets, float lr, float lambda1,
                             float lambda2_plus_beta_div_lr, float** state_tensors,
                             float** weight_tensors, float scaler, float* g) {
  const uint32_t start = ev_offsets[idx];
  const uint32_t end = ev_offsets[idx + 1];

  float* n = state_tensors[idx] - start;
  float* z = n + end - start;
  float* w = weight_tensors[idx] - start;

  for (uint32_t i = start; i < end; ++i) {
    float gi = g[i] / scaler;
    float ni = n[i];
    float ni_prev_sqrt = std::sqrt(ni);
    n[i] = ni = ni + gi * gi;
    float ni_sqrt = std::sqrt(ni);
    float sigma = (ni_sqrt - ni_prev_sqrt) / lr;
    float wi = w[i];
    float zi = z[i] = z[i] + gi - sigma * wi;

    float p = (1. - 2. * std::signbit(zi)) * lambda1 - zi;
    float q = ni_sqrt / lr + lambda2_plus_beta_div_lr;
    g[i] = (p / q) * std::signbit(lambda1 - std::abs(zi)) - wi;
  }
}

}  // namespace embedding