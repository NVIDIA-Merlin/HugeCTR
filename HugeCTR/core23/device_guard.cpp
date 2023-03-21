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

#include <core23/device_guard.hpp>
#include <core23/logger.hpp>

namespace HugeCTR {

namespace core23 {

DeviceGuard::DeviceGuard() : original_device_(Device::current()) {}
DeviceGuard::DeviceGuard(const Device& new_device) : DeviceGuard() {
  if (new_device != original_device_) {
    set_device(new_device);
  }
}

DeviceGuard::~DeviceGuard() { set_device(original_device_); }

void DeviceGuard::set_device(const Device& device) {
  if (device.type() != DeviceType::CPU) {
    HCTR_LIB_THROW(cudaSetDevice(device.index()));
  }
}

}  // namespace core23
}  // namespace HugeCTR
