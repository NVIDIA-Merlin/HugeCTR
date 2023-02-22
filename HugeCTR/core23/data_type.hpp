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

#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>

namespace HugeCTR {
namespace core23 {

#define ALL_DATA_TYPES_SUPPORTED(PH) \
  PH(__half, Half)                   \
  PH(float, Float)                   \
  PH(double, Double)                 \
  PH(char, Char)                     \
  PH(int8_t, Int8)                   \
  PH(uint8_t, UInt8)                 \
  PH(int32_t, Int32)                 \
  PH(uint32_t, UInt32)               \
  PH(int64_t, Int64)                 \
  PH(uint64_t, UInt64)

#define DATA_TYPE_ENUMERIZE(_, E) E,

enum class ScalarType : int8_t { ALL_DATA_TYPES_SUPPORTED(DATA_TYPE_ENUMERIZE) };

template <typename BuiltInType>
struct ToScalarType;

template <ScalarType ScalarType>
struct ToBuiltInType;

#define REGISTER_DATA_TYPE_CONVERTER(X, Y)             \
  template <>                                          \
  struct ToScalarType<X> {                             \
    constexpr static ScalarType value = ScalarType::Y; \
  };                                                   \
  template <>                                          \
  struct ToBuiltInType<ScalarType::Y> {                \
    using value = X;                                   \
  };

ALL_DATA_TYPES_SUPPORTED(REGISTER_DATA_TYPE_CONVERTER)

class DataType {
 public:
  DataType(ScalarType type) : type_(type) {}
  DataType() : DataType(ScalarType::Float) {}

  template <typename BuiltInType>
  bool match() const {
    return ToScalarType<BuiltInType>::value == type_;
  }

  int64_t size() const;

  std::string name() const;

  ScalarType type() const { return type_; }

  bool operator==(const DataType &other) const { return this->type() == other.type(); }
  bool operator!=(const DataType &other) const { return !(*this == other); }

 private:
  ScalarType type_;
};

std::ostream &operator<<(std::ostream &os, const DataType &d);

}  // namespace core23
}  // namespace HugeCTR

namespace std {
template <>
struct hash<HugeCTR::core23::DataType> {
  size_t operator()(HugeCTR::core23::DataType d) const {
    return std::hash<std::string>{}(d.name());
  }
};
}  // namespace std
