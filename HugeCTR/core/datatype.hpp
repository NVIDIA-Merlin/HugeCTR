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
#include <cuda_fp16.h>

#include <ostream>
#include <string>

namespace core {

enum class TensorScalarType : uint8_t {
  None = 0,
  Void,
  Float32,
  Float16,
  Int64,
  UInt64,
  Int32,
  UInt32,
  Size_t,
  Char,
  MAX_DATATYPE_META
};

constexpr size_t itemsize_map[static_cast<int>(TensorScalarType::MAX_DATATYPE_META)] = {
    1,                 // none
    sizeof(char),      // void
    sizeof(float),     // float32
    sizeof(__half),    // float16
    sizeof(int64_t),   // int64
    sizeof(uint64_t),  // uint64_t
    sizeof(int32_t),   // int32_t
    sizeof(uint32_t),  // uint32_t
    sizeof(size_t),    // size_t
    sizeof(char),      // char
};

std::ostream &operator<<(std::ostream &os, const TensorScalarType &t);

class DataType final {
  TensorScalarType type_;

 public:
  DataType() : type_(TensorScalarType::None) {}

  DataType(TensorScalarType type) : type_(type) {}

  template <typename T>
  bool match() const {
    return (std::is_same<float, T>::value && type_ == TensorScalarType::Float32) ||
           (std::is_same<__half, T>::value && type_ == TensorScalarType::Float16) ||
           (std::is_same<int64_t, T>::value && type_ == TensorScalarType::Int64) ||
           (std::is_same<uint64_t, T>::value && type_ == TensorScalarType::UInt64) ||
           (std::is_same<int32_t, T>::value && type_ == TensorScalarType::Int32) ||
           (std::is_same<uint32_t, T>::value && type_ == TensorScalarType::UInt32) ||
           (std::is_same<size_t, T>::value && type_ == TensorScalarType::Size_t) ||
           (std::is_same<char, T>::value && type_ == TensorScalarType::Char) ||
           type_ == TensorScalarType::Void;
  }

  size_t itemsize() const { return itemsize_map[static_cast<int>(type_)]; }

  TensorScalarType type() const { return type_; }

  bool operator==(const DataType &other) const { return this->type() == other.type(); }

  bool operator!=(const DataType &other) const { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &os, const DataType &d);
}  // namespace core

namespace std {
template <>
struct hash<core::DataType> {
  size_t operator()(core::DataType d) const {
    // TODO: move TensorScalarType from HugeCTR to core
    static_assert(sizeof(core::TensorScalarType) == 1, "TensorScalarType is not 8-bit");
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()));
    return std::hash<uint32_t>{}(bits);
  }
};
}  // namespace std
