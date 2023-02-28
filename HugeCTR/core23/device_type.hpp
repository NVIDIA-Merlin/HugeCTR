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

#include <string>

namespace HugeCTR {

namespace core23 {

#define ALL_DEVICE_TYPES_SUPPORTED(PH) \
  PH(CPU)                              \
  PH(GPU)                              \
  PH(UNIFIED)

#define ENUMERIZE(D) D,
enum class DeviceType : int8_t { ALL_DEVICE_TYPES_SUPPORTED(ENUMERIZE) };

std::string GetDeviceTypeName(DeviceType device_type);

std::ostream &operator<<(std::ostream &os, DeviceType device_type);

}  // namespace core23
}  // namespace HugeCTR
