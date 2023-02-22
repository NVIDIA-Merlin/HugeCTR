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

#include <base/debug/logger.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/buffer.hpp>
#include <core23/buffer_channel.hpp>
#include <core23/buffer_params.hpp>
#include <core23/details/buffer_factory_impl.hpp>
#include <core23/details/confederal_buffer.hpp>
#include <core23/details/unitary_buffer.hpp>
#include <core23/device.hpp>
#include <unordered_map>

namespace HugeCTR {

namespace core23 {

std::shared_ptr<Buffer> CreateBuffer(BufferParams buffer_params, const Device& device,
                                     std::unique_ptr<Allocator> allocator) {
  HCTR_THROW_IF(allocator == nullptr, HugeCTR::Error_t::IllegalCall,
                "A Buffer must be created but no allocator is specified.");

  std::shared_ptr<Buffer> buffer;
  if (buffer_params.custom_factory) {
    buffer = buffer_params.custom_factory(buffer_params, device, std::move(allocator));
  } else {
    if (buffer_params.unitary) {
      buffer = std::make_shared<UnitaryBuffer>(device, std::move(allocator));
    } else {
      buffer = std::make_shared<ConfederalBuffer>(device, std::move(allocator));
    }
  }

  return buffer;
}

}  // namespace core23

}  // namespace HugeCTR