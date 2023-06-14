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

#include <core23/allocator_factory.hpp>
#include <core23/buffer_factory.hpp>
#include <core23/buffer_params.hpp>
#include <core23/details/confederal_buffer.hpp>
#include <core23/details/unitary_buffer.hpp>
#include <core23/logger.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace HugeCTR {

namespace core23 {

namespace {

std::unordered_map<Device, std::unordered_set<BufferChannel>> g_buffers;

}  // namespace

std::shared_ptr<Buffer> GetBuffer(const BufferParams& buffer_params, const Device& device,
                                  std::unique_ptr<Allocator> allocator) {
  std::shared_ptr<Buffer> buffer;
  if (BufferParams::custom_factory) {
    buffer = BufferParams::custom_factory(buffer_params, device, std::move(allocator));
  } else {
    auto p_dev = g_buffers.insert({device, {}});

    auto& channels = p_dev.first->second;
    auto p_ch = channels.insert(buffer_params.channel);
    auto& channel = *p_ch.first;
    bool should_create = (p_ch.second || !channel.has_buffer({}));

    if (should_create) {
      HCTR_THROW_IF(allocator == nullptr, HugeCTR::Error_t::IllegalCall,
                    "A Buffer must be created but no allocator is specified.");

      if (buffer_params.unitary) {
        buffer = std::make_shared<UnitaryBuffer>(device, std::move(allocator));
      } else {
        buffer = std::make_shared<ConfederalBuffer>(device, std::move(allocator));
      }
      channel.set_buffer(buffer, {});
    }
    buffer = channel.get_buffer({});
  }
  return buffer;
}

bool AllocateBuffers(const Device& device) {
  auto it = g_buffers.find(device);
  if (it != g_buffers.end()) {
    for (auto& channel : it->second) {
      if (auto buffer = channel.get_buffer({}); buffer && buffer->allocatable()) {
        buffer->allocate();
      }
    }
    return true;
  }
  return false;
}

}  // namespace core23

}  // namespace HugeCTR
