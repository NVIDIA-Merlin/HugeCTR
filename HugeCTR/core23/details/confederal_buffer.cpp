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
#include <core23/allocator.hpp>
#include <core23/buffer_client.hpp>
#include <core23/details/confederal_buffer.hpp>
#include <core23/offsetted_buffer.hpp>

namespace HugeCTR {

namespace core23 {

ConfederalBuffer::ConfederalBuffer(std::unique_ptr<Allocator> allocator)
    : Buffer(std::move(allocator)), current_offset_(0LL) {}

ConfederalBuffer::~ConfederalBuffer() {
  for (auto [offset, ptr] : offset_to_ptr_) {
    allocator()->deallocate(ptr);
  }
}

Buffer::ClientOffsets ConfederalBuffer::do_allocate(const std::unique_ptr<Allocator>& allocator,
                                                    const ClientRequirements& client_requirements) {
  if (client_requirements.empty()) {
    HCTR_OWN_THROW(
        HugeCTR::Error_t::IllegalCall,
        "The buffer doesn't have any subscriber at all. What is the point of allocate()?");
  }

  ClientOffsets client_offsets;

  for (auto& [client, requirements] : client_requirements) {
    int64_t size = requirements.num_bytes;
    auto ptr = allocator->allocate(size, requirements.stream);
    if (ptr == nullptr) {
      HCTR_OWN_THROW(HugeCTR::Error_t::OutOfMemory,
                     "The ConfederalBuffer failed to allocate the memory");
    }

    client_offsets[client] = current_offset_;
    offset_to_ptr_[current_offset_] = ptr;
    current_offset_ += size;
  }

  return client_offsets;
}

void ConfederalBuffer::post_unsubscribe(const BufferClient* client,
                                        const BufferRequirements& requirements, int64_t offset) {
  auto it = offset_to_ptr_.find(offset);
  HCTR_THROW_IF(it == offset_to_ptr_.end(), HugeCTR::Error_t::IllegalCall,
                "post_unsubscribe() cannot be called without the memory allocation.");
  allocator()->deallocate(it->second, requirements.stream);
  offset_to_ptr_.erase(it);
}

}  // namespace core23

}  // namespace HugeCTR
