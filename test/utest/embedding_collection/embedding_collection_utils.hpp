/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <sys/time.h>

#include <numeric>

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "embedding_collection_cpu.hpp"

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
    float error = std::abs(a[i] - b[i]);
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
    float error = std::abs(lhs - rhs);
    ASSERT_LE(error, threshold) << ",lhs:" << lhs << ",rhs:" << rhs << "\n";
    max_error = std::max(max_error, error);
  }
  ASSERT_LE(max_error, threshold) << "max error:" << max_error << ",threshold:" << threshold;
}
