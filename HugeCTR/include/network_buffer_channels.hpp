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

#include <core23/buffer_channel_helpers.hpp>

namespace HugeCTR {

enum class NetworkBufferChannelType : int8_t {
  Blobs = 1,
  Weight = 2,
  WeightHalf = 3,
  Wgrad = 4,
  WgradHalf = 5,
  OptState = 6
};

std::string SetNetworkBufferChannel(NetworkBufferChannelType type, const std::string& new_name);
std::string GetNetworkBufferChannel(NetworkBufferChannelType type);
core23::BufferChannel GetBlobsBufferChannel();

core23::BufferChannel GetWeightBufferChannel();

core23::BufferChannel GetWeightHalfBufferChannel();

core23::BufferChannel GetWgradBufferChannel();

core23::BufferChannel GetWgradHalfBufferChannel();

core23::BufferChannel GetOptStateBufferChannnel();

}  // namespace HugeCTR
