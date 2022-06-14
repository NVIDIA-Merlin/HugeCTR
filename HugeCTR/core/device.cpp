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
#include "device.hpp"

#include <sstream>

namespace core {

inline std::string DeviceTypeName(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::GPU:
      return "cuda";
    case DeviceType::CPUGPU:
      return "cpugpu";
    default:
#ifndef TF_IMPL_UT
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "device type not initialized");
#else
      return "";
#endif
  }
  return "";
}

std::ostream &operator<<(std::ostream &os, DeviceType type) {
  os << DeviceTypeName(type);
  return os;
}

std::ostream &operator<<(std::ostream &os, Device device) {
  os << device.type();
  if (device.is_gpu()) {
    os << ":" << static_cast<int>(device.index());
  }
  return os;
}

}  // namespace core