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

#include <cuda_runtime.h>

#include <core23/buffer_client.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/data_type.hpp>
#include <core23/shape.hpp>
#include <core23/tensor_params.hpp>
#include <core23/tensor_view.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace HugeCTR {

namespace core23 {

class Device;

class TensorImpl : public BufferClient {
 public:
  TensorImpl(TensorParams params);

  TensorImpl(void* data, const Shape& shape, const DataType& data_type, const Device& device)
      : params_(TensorParams().data_type(data_type).shape(shape).device(device)),
        bound_data_(data) {}

  void* data() const;

  int64_t dims() const { return shape().dims(); }
  const Shape& shape() const { return params_.shape(); }
  int64_t size(int64_t dim) const { return shape().size(dim); }
  int64_t num_elements() const { return shape().size(); }
  int64_t num_bytes() const { return data_type().size() * num_elements(); }
  DataType data_type() const { return params_.data_type(); }
  const Device device() const { return params_.device(); }
  const TensorParams& my_params() const { return params_; }

  bool empty() const { return !own_data() && !bound_data_; }
  bool own_data() const { return offsetted_buffer() != nullptr; }

 private:
  TensorParams params_;
  std::optional<void*> bound_data_;
};

}  // namespace core23

}  // namespace HugeCTR