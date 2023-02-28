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

#include <base/debug/logger.hpp>
#include <core23/device.hpp>

namespace HugeCTR {

namespace core23 {

Device Device::current() {
  try {
    int raw_device_id;
    HCTR_LIB_THROW(cudaGetDevice(&raw_device_id));
    return Device(DeviceType::GPU, raw_device_id);
  } catch (...) {
    return Device(DeviceType::CPU);
  }
}

int64_t Device::count() {
  try {
    int count;
    HCTR_LIB_THROW(cudaGetDeviceCount(&count));
    return count;
  } catch (...) {
    return 0;
  }
}

Device::Device(DeviceType type, DeviceIndex index)
    : type_(type), index_(type == DeviceType::CPU ? 0 : index) {}

const std::string Device::name() const {
  return GetDeviceTypeName(type_) + ":" + std::to_string(index_);
}

std::ostream &operator<<(std::ostream &os, Device device) {
  os << device.name();
  return os;
}

}  // namespace core23
}  // namespace HugeCTR