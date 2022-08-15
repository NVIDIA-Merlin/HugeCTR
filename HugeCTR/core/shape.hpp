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
#include <stdint.h>

#include <iostream>
#include <vector>

#include "HugeCTR/include/base/debug/logger.hpp"

namespace core {

#if __cplusplus >= 201703L
template <typename T>
inline constexpr bool is_integer = std::is_integral_v<T> || std::is_unsigned_v<T>;
#endif

class Shape final {
  std::vector<int64_t> shape_;

 public:
  Shape() = default;

#if __cplusplus >= 201703L
  template <typename... T, typename = typename std::enable_if_t<(is_integer<T> && ...)> >
  constexpr Shape(T... s) {
    (shape_.push_back(static_cast<int64_t>(s)), ...);
  }

  template <typename T, typename = typename std::enable_if_t<(is_integer<T>)> >
  constexpr Shape(const std::vector<T> &s) {
    for (size_t i = 0; i < s.size(); ++i) {
      shape_.push_back(static_cast<int64_t>(s[i]));
    }
  }
#else
  template <typename T>
  constexpr Shape(const std::vector<T> &s) {
    for (size_t i = 0; i < s.size(); ++i) {
      shape_.push_back(static_cast<int64_t>(s[i]));
    }
  }
#endif

  int64_t operator[](size_t idx) const;

  size_t size() const { return shape_.size(); }

  int64_t num_elements() const;

  bool operator==(const Shape &other) const;

  bool operator!=(const Shape &other) const;

  std::string str() const;
};

std::ostream &operator<<(std::ostream &os, const Shape &s);
}  // namespace core