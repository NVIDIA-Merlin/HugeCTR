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

#include <core23/buffer_channel.hpp>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/device.hpp>
#include <cstdint>
#include <functional>
#include <memory>

namespace HugeCTR {

namespace core23 {

class Buffer;

struct BufferParams {
  using CustomFactory = std::function<std::shared_ptr<Buffer>(
      const BufferParams&, const Device& device, std::unique_ptr<Allocator>)>;

  BufferChannel channel = GetRandomBufferChannel();
  bool unitary = true;
  static CustomFactory custom_factory;
};

}  // namespace core23
}  // namespace HugeCTR