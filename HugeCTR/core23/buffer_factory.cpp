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
#include <core23/details/buffer_factory_impl.hpp>
#include <core23/details/confederal_buffer.hpp>
#include <core23/details/unitary_buffer.hpp>
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
  auto p_dev = g_buffers.insert({device, {}});
  bool should_create = p_dev.second;

  auto& channels = p_dev.first->second;
  auto p_ch = channels.insert(buffer_params.channel);
  auto& channel = *p_ch.first;
  should_create = (p_ch.second || !channel.has_buffer({}));

  std::shared_ptr<Buffer> buffer = nullptr;
  if (should_create) {
    buffer = CreateBuffer(buffer_params, device, std::move(allocator));
    channel.set_buffer(buffer, {});
  }
  return channel.get_buffer({});
}

}  // namespace core23

}  // namespace HugeCTR
