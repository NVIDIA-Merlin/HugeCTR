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

#include <cstdint>
#include <memory>

#pragma once

namespace HugeCTR {

namespace core23 {

class Buffer;

class OffsettedBuffer final {
 private:
  class ConstructKey {
    friend class Buffer;

   private:
    ConstructKey() {}
    ConstructKey(const ConstructKey&) = default;
  };

 public:
  OffsettedBuffer(const std::shared_ptr<Buffer>& buffer, ConstructKey)
      : buffer_(buffer), initialized_(false), offset_(0) {}
  OffsettedBuffer(const std::shared_ptr<Buffer>& buffer, int64_t offset, ConstructKey)
      : buffer_(buffer), initialized_(true), offset_(offset) {}
  ~OffsettedBuffer();

  void* data() const;

 private:
  std::shared_ptr<Buffer> buffer_;
  bool initialized_;
  int64_t offset_;
};

}  // namespace core23

}  // namespace HugeCTR
