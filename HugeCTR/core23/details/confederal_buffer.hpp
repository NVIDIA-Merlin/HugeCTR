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
#include <unordered_map>

namespace HugeCTR {

namespace core23 {

class BufferClient;
class OffsettedBuffer;

class ConfederalBuffer final : public Buffer {
 public:
  ConfederalBuffer(std::unique_ptr<Allocator> allocator);
  ~ConfederalBuffer() override;

  bool subscribable() const;

 private:
  using ClientRequirements = typename Buffer::ClientRequirements;

  void* data_impl(int64_t offset) const override { return offset_to_ptr_.find(offset)->second; }
  ClientOffsets do_allocate(const std::unique_ptr<Allocator>& allocator,
                            const ClientRequirements& client_requirements) override;

  void post_unsubscribe(const BufferClient* client, const BufferRequirements& requirements,
                        int64_t offset) override;

  std::unordered_map<int64_t, void*> offset_to_ptr_;
  int64_t current_offset_;
};

}  // namespace core23

}  // namespace HugeCTR
