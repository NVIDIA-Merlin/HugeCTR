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

template <typename T>
HugeCTR::Tensor2<T> wrap_tensor2_with_core23_tensor(const core23::Tensor& core23_tensor) {
  HCTR_CHECK_HINT(core23::ToScalarType<T>::value == core23_tensor.data_type().type(),
                  "cannot convert to different datatype ");
  auto shape = core23_tensor.shape();
  std::vector<size_t> dimensions(shape.data(), shape.data() + shape.dims());
  auto buffer = std::make_shared<Core23WrappingBuffer>(core23_tensor);
  return HugeCTR::Tensor2<T>(dimensions, buffer);
}

template <typename T>
HugeCTR::Tensors2<T> wrap_tensors2_with_core23_tensors(
    const std::vector<core23::Tensor>& core23_tensors) {
  HugeCTR::Tensors2<T> tensors2;
  std::transform(core23_tensors.begin(), core23_tensors.end(), std::back_inserter(tensors2),
                 [](const core23::Tensor& t) { return wrap_tensor2_with_core23_tensor<T>(t); });
  return tensors2;
}

template <typename T>
core23::Tensor wrap_core23_tensor_with_tensor2(Tensor2<T> tensor2, const core23::Device& device) {
  std::vector<int64_t> dimensions;
  std::transform(tensor2.get_dimensions().begin(), tensor2.get_dimensions().end(),
                 std::back_inserter(dimensions),
                 [](const size_t& d) { return static_cast<int64_t>(d); });
  core23::Shape shape(dimensions);
  core23::BufferParams buffer_params;
  buffer_params.custom_factory = [tensor2](const auto&, const auto& device, auto allocator) {
    return std::make_shared<LegacyWrappingBuffer<T>>(tensor2, device, std::move(allocator));
  };
  return core23::Tensor(core23::TensorParams(shape).device(device).buffer_params(buffer_params));
}
template <typename T>
std::vector<core23::Tensor> wrap_core23_tensors_with_tensors2(const Tensors2<T>& tensors2,
                                                              const core23::Device& device) {
  std::vector<core23::Tensor> core23_tensors;
  std::transform(tensors2.begin(), tensors2.end(), std::back_inserter(core23_tensors),
                 [device](const auto& t) { return wrap_core23_tensor_with_tensor2<T>(t, device); });
  return core23_tensors;
}

}  // namespace HugeCTR