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

#include <optional>
#include <string>

#include <core23/device.hpp>

namespace HugeCTR {
namespace core23 {

class Device;

class DeviceGuard final {
 public:
  DeviceGuard();
  DeviceGuard(const Device& device);
  ~DeviceGuard();

  void set_device(const Device& device);

 private:
  Device original_device_;
};

}  // namespace core23
}  // namespace HugeCTR
