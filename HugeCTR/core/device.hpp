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
#include <ostream>
#include <string>

namespace core {

enum class DeviceType : int8_t {
  CPU = 0,
  GPU = 1,
  CPUGPU = 2,
  MAX_DEVICE_NUM = 3,
};

std::ostream &operator<<(std::ostream &os, DeviceType type);

using DeviceIndex = int8_t;

class Device final {
  DeviceType type_;
  DeviceIndex index_;  // currently only for device type GPU/CPUGPU

 public:
  Device() : type_(DeviceType::MAX_DEVICE_NUM), index_(-1) {}

  Device(DeviceType type, int index = -1);

  bool operator==(const Device &other) const {
    return (type_ == other.type()) && (index_ == other.index());
  }

  bool operator!=(const Device &other) const { return !(*this == other); }

  DeviceType type() const { return type_; }

  DeviceIndex index() const { return index_; }

  bool is_gpu() const { return (type_ == DeviceType::GPU) || (type_ == DeviceType::CPUGPU); }
};

std::ostream &operator<<(std::ostream &os, Device device);

}  // namespace core

namespace std {
template <>
struct hash<core::Device> {
  size_t operator()(core::Device d) const {
    static_assert(sizeof(core::DeviceType) == 1, "DeviceType is not 8-bit");
    static_assert(sizeof(core::DeviceIndex) == 1, "DeviceIndex is not 8-bit");
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type())) << 16 |
                    static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
}  // namespace std