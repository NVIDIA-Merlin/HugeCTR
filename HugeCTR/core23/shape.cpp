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

#include <core23/shape.hpp>
#include <cstdint>
#include <vector>

namespace HugeCTR {

namespace core23 {

bool Shape::operator==(const Shape &other) const {
  return dims() == other.dims() && [&] {
    for (size_t i = 0; i < dims(); ++i) {
      if (size(i) != other.size(i)) {
        return false;
      }
    }
    return true;
  }();
}

std::string Shape::str() const {
  std::string str = "(";
  for (int64_t dim = 0; dim < dims(); dim++) {
    str += std::to_string(size(dim));
    str += (dim == dims() - 1) ? ")" : ", ";
  }
  return str;
}

std::ostream &operator<<(std::ostream &os, const Shape &s) {
  os << s.str();
  return os;
}

}  // namespace core23

}  // namespace HugeCTR
