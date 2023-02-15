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

#include <algorithm>
#include <core23/details/tensor_helpers.hpp>
#include <core23/macros.hpp>

namespace HugeCTR {

namespace core23 {

template <typename BuiltInType, int64_t Dims>
class TensorViewImplBase {
 public:
  HCTR_HOST_DEVICE TensorViewImplBase(BuiltInType* data, const int64_t* shape,
                                      const int64_t* strides, const int64_t* offsets)
      : data_(data), shape_(shape), strides_(strides), offsets_(offsets) {}

 protected:
  BuiltInType* data_;
  const int64_t* shape_;
  const int64_t* strides_;
  const int64_t* offsets_;
};

template <typename BuiltInType, int64_t Dims>
class TensorViewImpl : public TensorViewImplBase<BuiltInType, Dims> {
 public:
  HCTR_HOST_DEVICE TensorViewImpl(BuiltInType* data, const int64_t* shape, const int64_t* strides,
                                  const int64_t* offsets)
      : TensorViewImplBase<BuiltInType, Dims>(data, shape, strides, offsets) {}

  HCTR_HOST_DEVICE TensorViewImpl<BuiltInType, Dims - 1> operator[](int64_t index) const {
    return TensorViewImpl<BuiltInType, Dims - 1>(
        this->data_ + this->strides_[0] * (index + this->offsets_[0]), this->shape_ + 1,
        this->strides_ + 1, this->offsets_ + 1);
  }
};

template <typename BuiltInType>
class TensorViewImpl<BuiltInType, 1> : public TensorViewImplBase<BuiltInType, 1> {
 public:
  HCTR_HOST_DEVICE TensorViewImpl(BuiltInType* data, const int64_t* shape, const int64_t* strides,
                                  const int64_t* offsets)
      : TensorViewImplBase<BuiltInType, 1>(data, shape, strides, offsets) {}

  HCTR_HOST_DEVICE BuiltInType& operator[](int64_t index) const {
    return this->data_[this->strides_[0] * (index + this->offsets_[0])];
  }
};

template <typename BuiltInType, int64_t Dims>
class TensorViewBase {
 public:
  HCTR_HOST TensorViewBase(BuiltInType* data, const int64_t* shape, const int64_t* offsets)
      : data_(data) {
    std::copy(shape, shape + Dims, shape_);
    compute_strides(shape_, strides_);
    std::copy(offsets, offsets + Dims, offsets_);
    for (int64_t d = 0; d < Dims; d++) {
      shape_[d] -= offsets[d];
    }
  }
  HCTR_HOST TensorViewBase(BuiltInType* data, const int64_t* shape) : data_(data) {
    std::copy(shape, shape + Dims, shape_);
    compute_strides(shape_, strides_);
    std::fill(offsets_, offsets_ + Dims, 0);
  }
  HCTR_INLINE HCTR_HOST_DEVICE int64_t dims() const { return Dims; }
  HCTR_INLINE HCTR_HOST_DEVICE int64_t size(int64_t dim) const { return shape_[dim]; }
  HCTR_INLINE HCTR_HOST_DEVICE int64_t stride(int64_t dim) const { return strides_[dim]; }
  HCTR_INLINE HCTR_HOST_DEVICE int64_t offset(int64_t dim) const { return offsets_[dim]; }
  HCTR_INLINE HCTR_HOST_DEVICE BuiltInType* data() const { return data_; }

 protected:
  BuiltInType* data_;
  int64_t shape_[Dims];
  int64_t strides_[Dims];
  int64_t offsets_[Dims];
};

template <typename BuiltInType, int64_t Dims>
class TensorView : public TensorViewBase<BuiltInType, Dims> {
 public:
  HCTR_HOST TensorView(BuiltInType* data, const int64_t* shape, const int64_t* offsets)
      : TensorViewBase<BuiltInType, Dims>(data, shape, offsets) {}
  HCTR_HOST TensorView(BuiltInType* data, const int64_t* shape)
      : TensorViewBase<BuiltInType, Dims>(data, shape) {}

  HCTR_INLINE HCTR_HOST_DEVICE TensorViewImpl<BuiltInType, Dims - 1> operator[](
      int64_t index) const {
    return TensorViewImpl<BuiltInType, Dims - 1>(
        this->data_ + this->strides_[0] * (index + this->offsets_[0]), this->shape_ + 1,
        this->strides_ + 1, this->offsets_ + 1);
  }
};

template <typename BuiltInType>
class TensorView<BuiltInType, 1> : public TensorViewBase<BuiltInType, 1> {
 public:
  HCTR_HOST TensorView(BuiltInType* data, const int64_t* shape, const int64_t* offsets)
      : TensorViewBase<BuiltInType, 1>(data, shape, offsets) {}
  HCTR_HOST TensorView(BuiltInType* data, const int64_t* shape)
      : TensorViewBase<BuiltInType, 1>(data, shape) {}

  HCTR_INLINE HCTR_HOST_DEVICE BuiltInType& operator[](int64_t index) const {
    return this->data_[this->strides_[0] * (index + this->offsets_[0])];
  }
};

}  // namespace core23

}  // namespace HugeCTR