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

#include <core23/device_type.hpp>
#include <string>

namespace HugeCTR {
namespace core23 {

using DeviceIndex = int8_t;

class Device final {
 public:
  static Device current();
  static int64_t count();
  Device(DeviceType type, DeviceIndex index = 0);
  Device() : Device(DeviceType::GPU, 0) {}

  const std::string name() const;

  bool operator==(const Device& other) const {
    return (type_ == other.type()) && (index_ == other.index());
  }
  bool operator!=(const Device& other) const { return !(*this == other); }

  DeviceType type() const { return type_; }
  DeviceIndex index() const { return index_; }

 private:
  DeviceType type_;
  DeviceIndex index_;
};

std::ostream& operator<<(std::ostream& os, Device device);

}  // namespace core23
}  // namespace HugeCTR

namespace std {
template <>
struct hash<HugeCTR::core23::Device> {
  size_t operator()(HugeCTR::core23::Device d) const {
    static_assert(sizeof(HugeCTR::core23::DeviceType) == 1, "DeviceType is not 8-bit");
    static_assert(sizeof(HugeCTR::core23::DeviceIndex) == 1, "DeviceIndex is not 8-bit");
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type())) << 16 |
                    static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
}  // namespace std
