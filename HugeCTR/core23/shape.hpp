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

#include <cstdint>
#include <string>
#include <vector>

namespace HugeCTR {

namespace core23 {

class Shape : private std::vector<int64_t> {
 public:
  Shape() {}
  Shape(int64_t dims) : std::vector<int64_t>(dims, 0) {}
  Shape(std::initializer_list<int64_t> l) : std::vector<int64_t>(l) {}
  Shape(const std::vector<int64_t> &l) : std::vector<int64_t>(l) {}
  bool valid() const { return size() != 0; }
  int64_t dims() const { return std::vector<int64_t>::size(); }
  int64_t size(int64_t dim) const { return empty() ? 0 : at(dim); }
  int64_t &operator[](const int64_t dim) { return at(dim); }
  const int64_t &operator[](const int64_t dim) const { return at(dim); }
  int64_t size() const {
    if (empty()) {
      return 0;
    }

    int64_t sum = 1;
    for (auto dim = 0; dim < dims(); dim++) {
      sum *= size(dim);
    }
    return sum;
  }
  void set(int64_t dim, int64_t size) { at(dim) = size; }
  const int64_t *data() const { return std::vector<int64_t>::data(); }

  bool operator==(const Shape &other) const;
  bool operator!=(const Shape &other) const { return !(*this == other); }

  std::string str() const;
};

std::ostream &operator<<(std::ostream &os, const Shape &s);

}  // namespace core23

}  // namespace HugeCTR
