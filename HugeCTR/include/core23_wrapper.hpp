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
#include <core23/buffer.hpp>
#include <core23/tensor.hpp>
#include <general_buffer2.hpp>
namespace HugeCTR {

// TODO: remove those wrapper after we deprecate hybrid embedding
class Core23WrappingBuffer : public TensorBuffer2 {
 public:
  Core23WrappingBuffer(const core23::Tensor& core23_tensor) : core23_tensor_(core23_tensor) {}
  ~Core23WrappingBuffer() override {}
  bool allocated() const override { return core23_tensor_.data() != nullptr; }
  void* get_ptr() override { return core23_tensor_.data(); }

 private:
  core23::Tensor core23_tensor_;
};

template <typename T>
class LegacyWrappingBuffer : public core23::Buffer {
 public:
  LegacyWrappingBuffer(HugeCTR::Tensor2<T> tensor2, const core23::Device& device,
                       std::unique_ptr<core23::Allocator> allocator)
      : core23::Buffer(device, std::move(allocator)), tensor2_(tensor2) {}
  ~LegacyWrappingBuffer() override {}

 private:
  void* data_impl(int64_t offset) const override { return const_cast<float*>(tensor2_.get_ptr()); }
  core23::Buffer::ClientOffsets do_allocate(
      const std::unique_ptr<core23::Allocator>& allocator,
      const ClientRequirements& client_requirements) override {
    ClientOffsets client_offsets;
    int64_t index = 0LL;
    for (auto& [client, requirements] : client_requirements) {
      client_offsets[client] = index++;
    }
    return client_offsets;
  }

  HugeCTR::Tensor2<T> tensor2_;
};

}  // namespace HugeCTR