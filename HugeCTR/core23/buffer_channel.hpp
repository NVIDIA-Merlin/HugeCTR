/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <functional>
#include <iostream>
#include <memory>

namespace HugeCTR {
namespace core23 {

class Buffer;
struct BufferParams;
class Allocator;
class Device;

class BufferChannel {
 private:
  class Key {
    friend std::shared_ptr<Buffer> GetBuffer(const BufferParams& buffer_params,
                                             const Device& device,
                                             std::unique_ptr<Allocator> allocator);

   private:
    Key() {}
    Key(const Key&) = default;
  };

 public:
  BufferChannel(const std::string& name) : raw_channel_(std::hash<std::string>{}(name)) {}
  BufferChannel(const char* name) : BufferChannel(std::string(name)) {}
  bool operator==(const BufferChannel& other) const { return (raw_channel_ == other.raw_channel_); }
  bool operator!=(const BufferChannel& other) const { return !(*this == other); }
  operator int64_t() const noexcept { return raw_channel_; }
  int64_t operator()() const noexcept { return *this; }

  bool has_buffer(Key) const { return !buffer_.expired(); }
  void set_buffer(std::shared_ptr<Buffer> buffer, Key) const { buffer_ = buffer; }
  std::shared_ptr<Buffer> get_buffer(Key) const { return buffer_.lock(); }

 private:
  int64_t raw_channel_;
  mutable std::weak_ptr<Buffer> buffer_;
};

std::ostream& operator<<(std::ostream& os, BufferChannel buffer_channel);

}  // namespace core23
}  // namespace HugeCTR

namespace std {
template <>
struct hash<HugeCTR::core23::BufferChannel> {
  size_t operator()(HugeCTR::core23::BufferChannel ch) const { return std::hash<int64_t>{}(ch); }
};
}  // namespace std