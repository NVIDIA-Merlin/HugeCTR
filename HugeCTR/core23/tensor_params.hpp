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

#include <base/debug/logger.hpp>
#include <core23/allocator_params.hpp>
#include <core23/buffer_params.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/data_type.hpp>
#include <core23/device.hpp>
#include <core23/shape.hpp>
#include <cstdint>
#include <vector>

namespace HugeCTR {

namespace core23 {

class TensorParams final {
 public:
  TensorParams(const Shape& shape, const DataType& data_type, int64_t alignment,
               const Device& device, const AllocatorParams allocator_params,
               const BufferParams& buffer_params, CUDAStream stream)
      : shape_(shape),
        data_type_(data_type),
        alignment_(alignment),
        device_(device),
        allocator_params_(allocator_params),
        buffer_params_(buffer_params),
        stream_(stream),
        has_shape_(shape.valid()) {}

  TensorParams() : TensorParams(Shape(), DataType(), 256, Device(), {}, {}, CUDAStream()) {}
  TensorParams(const Shape& shape) : TensorParams() { this->set_shape(shape); }

  TensorParams shape(const Shape& shape) const noexcept {
    TensorParams p = *this;
    p.set_shape(shape);
    return p;
  }
  TensorParams data_type(DataType data_type) const noexcept {
    TensorParams p = *this;
    p.data_type_ = data_type;
    return p;
  }
  TensorParams alignment(int64_t alignment) const noexcept {
    TensorParams p = *this;
    p.alignment_ = alignment;
    return p;
  }

  TensorParams device(const Device& device) const noexcept {
    TensorParams p = *this;
    p.device_ = device;
    return p;
  }

  TensorParams allocator_params(const AllocatorParams& allocator_params) const noexcept {
    TensorParams p = *this;
    p.allocator_params_ = allocator_params;
    return p;
  }
  TensorParams buffer_params(const BufferParams& buffer_params) const noexcept {
    TensorParams p = *this;
    p.buffer_params_ = buffer_params;
    return p;
  }

  TensorParams buffer_channel(const BufferChannel& buffer_channel) const noexcept {
    TensorParams p = *this;
    p.buffer_params_.channel = buffer_channel;
    return p;
  }

  TensorParams stream(CUDAStream& stream) const noexcept {
    TensorParams p = *this;
    p.stream_ = stream;
    return p;
  }

  const Shape& shape() const {
    HCTR_THROW_IF(shape_.valid() == false, HugeCTR::Error_t::IllegalCall,
                  "TensorParams's Shape is not valid yet");
    return shape_;
  };

  DataType data_type() const { return data_type_; }
  int64_t alignment() const { return alignment_; }
  Device device() const { return device_; }
  const AllocatorParams& allocator_params() const { return allocator_params_; }
  const BufferParams& buffer_params() const { return buffer_params_; }
  const BufferChannel& buffer_channel() const { return buffer_params_.channel; }
  CUDAStream stream() const { return stream_; }

 private:
  void set_shape(const Shape& shape) {
    shape_ = shape;
    has_shape_ = shape_.valid() ? true : false;
  }

  Shape shape_;
  DataType data_type_;
  int64_t alignment_;

  Device device_;
  AllocatorParams allocator_params_;
  BufferParams buffer_params_;
  CUDAStream stream_;

  bool has_shape_;
};

}  // namespace core23
}  // namespace HugeCTR