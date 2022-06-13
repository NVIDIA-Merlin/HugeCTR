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
#include "shape.hpp"

#include <sstream>

namespace core {

bool Shape::operator==(const Shape &other) const {
  return size() == other.size() && [&] {
    for (size_t i = 0; i < other.size(); ++i) {
      if (shape_[i] != other[i]) {
        return false;
      }
    }
    return true;
  }();
}

bool Shape::operator!=(const Shape &other) const { return !(*this == other); }

std::string Shape::str() const {
  std::stringstream ss;
  ss << "(";
  int64_t dim = static_cast<int64_t>(shape_.size());
  for (int i = 0; i < dim - 1; ++i) {
    ss << shape_[i] << ",";
  }
  if (dim > 0) {
    ss << shape_[dim - 1];
  }
  ss << ")";
  return ss.str();
}

std::ostream &operator<<(std::ostream &os, const Shape &s) {
  os << s.str();
  return os;
}
}  // namespace core