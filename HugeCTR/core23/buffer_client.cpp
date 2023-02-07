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
#include <core23/buffer_client.hpp>
#include <core23/offsetted_buffer.hpp>
#include <iostream>

namespace HugeCTR {

namespace core23 {

BufferClient::~BufferClient() {}

void BufferClient::on_subscribe(const std::shared_ptr<OffsettedBuffer>& offsetted_buffer) {
  offsetted_buffer_ = offsetted_buffer;
}

void BufferClient::on_unsubscribe(const OffsettedBuffer& offsetted_buffer) {
  *offsetted_buffer_ = offsetted_buffer;
}

void BufferClient::on_allocate(const OffsettedBuffer& offsetted_buffer) {
  *offsetted_buffer_ = offsetted_buffer;
}

}  // namespace core23

}  // namespace HugeCTR
