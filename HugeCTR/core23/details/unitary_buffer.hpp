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

#include <core23/buffer.hpp>
#include <queue>

namespace HugeCTR {

namespace core23 {

class BufferClient;
class OffsettedBuffer;

class UnitaryBuffer final : public Buffer {
 public:
  UnitaryBuffer(const Device& device, std::unique_ptr<Allocator> allocator);
  ~UnitaryBuffer() override;

  bool subscribable() const;

 private:
  using ClientRequirements = typename Buffer::ClientRequirements;

  void* data_impl(int64_t offset) const override {
    return static_cast<void*>(static_cast<char*>(ptr_) + offset);
  }
  ClientOffsets do_allocate(const std::unique_ptr<Allocator>& allocator,
                            const ClientRequirements& client_requirements) override;
  bool subscribable_impl() const override { return !allocated_; }
  bool allocatable_impl() const override { return subscribable_impl(); }

  void post_subscribe(const BufferClient* client, BufferRequirements requirements) override;

  int64_t compute_offset(int64_t offset, int64_t alignment);

  bool allocated_;
  void* ptr_;
  std::queue<BufferClient*> new_insertion_order_;
};

}  // namespace core23

}  // namespace HugeCTR
