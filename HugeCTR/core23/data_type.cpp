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

#include <core23/data_type.hpp>

namespace HugeCTR {
namespace core23 {

namespace {

#define DATA_TYPEOF(T, _) sizeof(T),
#define DATA_NAMEOF(_, E) #E,

constexpr int64_t scalar_type_size[] = {ALL_DATA_TYPES_SUPPORTED(DATA_TYPEOF)};

constexpr const char *scalar_type_name[] = {ALL_DATA_TYPES_SUPPORTED(DATA_NAMEOF)};

}  // namespace

int64_t DataType::size() const {
  return scalar_type_size[static_cast<std::underlying_type_t<ScalarType>>(type_)];
}

std::string DataType::name() const {
  return scalar_type_name[static_cast<std::underlying_type_t<ScalarType>>(type_)];
}

std::ostream &operator<<(std::ostream &os, const DataType &d) { return os << d.name(); }

}  // namespace core23
}  // namespace HugeCTR
