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

#include <cuda_runtime.h>

#include <core23/tensor_params.hpp>
#include <core23/tensor_view.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace HugeCTR {

namespace core23 {

class TensorImpl;
class Device;
class DataType;
class Shape;
struct TensorParams;

class Tensor {
 public:
  Tensor();
  Tensor(TensorParams params);

  Tensor(const Shape& shape, DataType data_type, TensorParams params = TensorParams());

  Tensor(const Tensor& rhs) = default;

  Tensor(Tensor&& rhs) = default;

  Tensor& operator=(Tensor rhs) {
    swap(rhs);
    return *this;
  }

  Tensor reshape(const Shape& new_shape);

  static Tensor bind(void* data, const Shape& shape, const DataType& data_type,
                     const Device& device);

  void* data() const;
  template <typename ScalarType>
  ScalarType* data() const {
    return static_cast<ScalarType*>(data());
  }

  int64_t dims() const;
  const Shape& shape() const;
  int64_t size(size_t dim) const;
  int64_t num_elements() const;
  int64_t num_bytes() const;
  DataType data_type() const;
  const Device device() const;

  TensorParams my_params() const;

  bool empty() const;
  bool is_unique() const;
  bool own_data() const;

  template <typename BuiltInType, int64_t Dims>
  TensorView<BuiltInType, Dims> view() const {
    return view<BuiltInType, Dims>(Shape(Dims));
  }

  template <typename BuiltInType, int64_t Dims>
  TensorView<BuiltInType, Dims> view(const Shape& offsets) const {
    HCTR_THROW_IF(!data_type().match<BuiltInType>(), HugeCTR::Error_t::IllegalCall,
                  "The specified `BuiltInType` is not consistent with Tensor's data_type");
    HCTR_THROW_IF(Dims > dims(), HugeCTR::Error_t::IllegalCall,
                  "`Dims` is inconsistent with the Tensor's shape.");

    int64_t shape_data[Dims] = {};
    int64_t accum = 1;
    for (int64_t d = dims() - 1; d >= 0; d--) {
      auto s = shape().size(d);
      if (d >= Dims) {
        accum *= s;
      } else {
        shape_data[d] = (d == Dims - 1) ? accum * s : s;
      }
    }

    return TensorView<BuiltInType, Dims>(static_cast<BuiltInType*>(data()), shape_data,
                                         offsets.data());
  }

 private:
  Tensor(const std::shared_ptr<TensorImpl>& impl);
  void swap(Tensor& rhs);

  std::shared_ptr<TensorImpl> impl_;

  Shape shape_or_;

  mutable void* data_;
};

}  // namespace core23

}  // namespace HugeCTR