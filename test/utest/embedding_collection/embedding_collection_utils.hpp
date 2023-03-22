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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/embedding.hpp>
#include <numeric>
#include <resource_managers/resource_manager_ext.hpp>

using namespace embedding;

template <typename type>
inline void print_array(size_t num, const std::vector<type> &a) {
  ASSERT_GE(a.size(), num);
  for (size_t i = 0; i < num; ++i) {
    std::cout << a[i] << " ";
  }
  std::cout << "\n";
}

template <>
inline void print_array<__half>(size_t num, const std::vector<__half> &a) {
  ASSERT_GE(a.size(), num);
  for (size_t i = 0; i < num; ++i) {
    std::cout << HugeCTR::TypeConvert<float, __half>::convert(a[i]) << " ";
  }
  std::cout << "\n";
}

template <typename type, typename = std::enable_if_t<std::is_integral_v<type>>>
inline void assert_array_eq(size_t num, const std::vector<type> &a, const std::vector<type> &b) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  for (size_t i = 0; i < num; ++i) {
    ASSERT_EQ(a[i], b[i]) << "idx:" << i;
  }
}

inline void assert_array_eq(size_t num, const std::vector<float> &a, const std::vector<float> &b,
                            float threshold = 1e-3) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  float max_error = 0.f;
  for (size_t i = 0; i < num; ++i) {
    float error;
    if (std::abs(a[i]) > 10.0f) {
      error = std::abs(a[i] - b[i]) / std::abs(a[i]);
    } else {
      error = std::abs(a[i] - b[i]);
    }

    max_error = std::max(max_error, error);
  }
  ASSERT_LE(max_error, threshold) << "max error:" << max_error << ",threshold:" << threshold;
}

inline void assert_array_eq(size_t num, const std::vector<__half> &a, const std::vector<__half> &b,
                            float threshold = 1e-3) {
  ASSERT_GE(a.size(), num);
  ASSERT_GE(b.size(), num);
  float max_error = 0.f;
  for (size_t i = 0; i < num; ++i) {
    float lhs = HugeCTR::TypeConvert<float, __half>::convert(a[i]);
    float rhs = HugeCTR::TypeConvert<float, __half>::convert(b[i]);
    float error;
    if (std::abs(lhs) > 10.0f) {
      error = std::abs(lhs - rhs) / std::abs(lhs);
    } else {
      error = std::abs(lhs - rhs);
    }

    ASSERT_LE(error, threshold) << ",lhs:" << lhs << ",rhs:" << rhs << "\n";
    max_error = std::max(max_error, error);
  }
  ASSERT_LE(max_error, threshold) << "max error:" << max_error << ",threshold:" << threshold;
}

inline float kahanSum(const std::vector<float> &fa) {
  float sum = 0.0;

  // Variable to store the error
  float c = 0.0;

  // Loop to iterate over the array
  for (float f : fa) {
    float y = f - c;
    float t = sum + y;

    // Algebraically, c is always 0
    // when t is replaced by its
    // value from the above expression.
    // But, when there is a loss,
    // the higher-order y is cancelled
    // out by subtracting y from c and
    // all that remains is the
    // lower-order error in c
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}
