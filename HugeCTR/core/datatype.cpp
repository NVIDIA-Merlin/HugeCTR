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

#include <common.hpp>
#include <core/datatype.hpp>

namespace core {

inline std::string TensorScalarTypeName(TensorScalarType t) {
  switch (t) {
    case TensorScalarType::None:
      return "none";
    case TensorScalarType::Void:
      return "void";
    case TensorScalarType::Float32:
      return "float32";
    case TensorScalarType::Float16:
      return "half";
    case TensorScalarType::Int64:
      return "int64";
    case TensorScalarType::UInt64:
      return "uint64";
    case TensorScalarType::Int32:
      return "int32";
    case TensorScalarType::UInt32:
      return "uint32";
    case TensorScalarType::Size_t:
      return "size_t";
    case TensorScalarType::Char:
      return "char";
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "data type not initialized");
  }
  return "";
}

std::ostream &operator<<(std::ostream &os, const TensorScalarType &t) {
  os << TensorScalarTypeName(t);
  return os;
}

std::ostream &operator<<(std::ostream &os, const DataType &d) {
  os << d.type();
  return os;
}

}  // namespace core