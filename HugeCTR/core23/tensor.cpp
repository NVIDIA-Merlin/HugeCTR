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

#include <core23/cuda_stream.hpp>
#include <core23/details/tensor_impl.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>

namespace HugeCTR {

namespace core23 {

Tensor::Tensor() : impl_(nullptr), data_(nullptr) {}

Tensor::Tensor(TensorParams params) : impl_(std::make_shared<TensorImpl>(params)), data_(nullptr) {}

Tensor::Tensor(const Shape& shape, DataType data_type, TensorParams params)
    : Tensor(params.shape(shape).data_type(data_type)) {}

Tensor Tensor::reshape(const Shape& new_shape) {
  HCTR_THROW_IF(new_shape.size() != num_elements(), HugeCTR::Error_t::IllegalCall,
                "The new shape doesn't have the same number of elements with the original shape.");

  Tensor reshaped_tensor(*this);
  reshaped_tensor.shape_or_ = new_shape;
  return reshaped_tensor;
}

Tensor Tensor::bind(void* data, const Shape& shape, const DataType& data_type,
                    const Device& device) {
  return Tensor(std::make_shared<TensorImpl>(data, shape, data_type, device));
}

void* Tensor::data() const { return data_ ? data_ : data_ = (impl_ ? impl_->data() : nullptr); }

int64_t Tensor::dims() const { return shape().dims(); }
const Shape& Tensor::shape() const {
  HCTR_THROW_IF(empty(), HugeCTR::Error_t::IllegalCall, "An empty Tensor doesn't have the shape");

  if (shape_or_.valid()) {
    return shape_or_;
  }
  return impl_->shape();
}  // namespace core23
int64_t Tensor::size(size_t dim) const { return shape().size(dim); }
int64_t Tensor::num_elements() const { return shape().size(); }
int64_t Tensor::num_bytes() const { return data_type().size() * num_elements(); }
const Device Tensor::device() const {
  HCTR_THROW_IF(empty(), HugeCTR::Error_t::IllegalCall,
                "An empty Tensor doesn't belong to any device");
  return impl_->device();
}
DataType Tensor::data_type() const {
  HCTR_THROW_IF(empty(), HugeCTR::Error_t::IllegalCall,
                "An empty Tensor doesn't have the data type");
  return impl_->data_type();
}

TensorParams Tensor::my_params() const {
  HCTR_THROW_IF(empty(), HugeCTR::Error_t::IllegalCall, "An empty Tensor doesn't have the params");
  return impl_->my_params().shape(shape());
}

bool Tensor::empty() const { return impl_ == nullptr || impl_->empty(); }

bool Tensor::is_unique() const { return own_data() && impl_.use_count() == 1; }

bool Tensor::own_data() const { return impl_ && impl_->own_data(); }

Tensor::Tensor(const std::shared_ptr<TensorImpl>& impl) : impl_(impl), data_(nullptr) {}

void Tensor::swap(Tensor& rhs) {
  std::swap(impl_, rhs.impl_);
  std::swap(shape_or_, rhs.shape_or_);
  std::swap(data_, rhs.data_);
}

}  // namespace core23

}  // namespace HugeCTR
