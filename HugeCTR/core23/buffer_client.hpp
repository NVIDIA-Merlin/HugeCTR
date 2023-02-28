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

#include <memory>

namespace HugeCTR {

namespace core23 {

class OffsettedBuffer;

class BufferClient {
 public:
  virtual ~BufferClient() = 0;

  void on_subscribe(const std::shared_ptr<OffsettedBuffer>& offsetted_buffer);
  void on_unsubscribe(const OffsettedBuffer& offsetted_buffer);
  void on_allocate(const OffsettedBuffer& offsetted_buffer);

 protected:
  const std::shared_ptr<OffsettedBuffer>& offsetted_buffer() const { return offsetted_buffer_; }

 private:
  std::shared_ptr<OffsettedBuffer> offsetted_buffer_;
};

}  // namespace core23

}  // namespace HugeCTR
