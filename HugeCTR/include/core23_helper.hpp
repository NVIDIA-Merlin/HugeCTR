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

#include <common.hpp>
#include <core23/tensor.hpp>
#include <inference/preallocated_buffer2.hpp>
#include <sparse_tensor.hpp>
#include <tensor2.hpp>
namespace HugeCTR {
namespace core_helper {
template <typename T>
HugeCTR::Tensor2<T> convert_core23_tensor_to_tensor2(core23::Tensor native_tensor) {
  HCTR_CHECK_HINT(core23::ToScalarType<T>::value == native_tensor.data_type().type(),
                  "cannot convert to different datatype ");
  auto shape = native_tensor.shape();
  std::vector<size_t> dimensions(shape.data(), shape.data() + shape.dims());
  auto buffer = PreallocatedBuffer2<T>::create(native_tensor.data(), dimensions);
  HugeCTR::Tensor2<T> tensor2(dimensions, buffer);
  return tensor2;
}
template <typename T>
HugeCTR::TensorBag2 convert_core23_tensor_to_tensorbag2(core23::Tensor native_tensor) {
  return convert_core23_tensor_to_tensor2<T>(native_tensor).shrink();
}

template <typename T>
core23::Tensor convert_tensor2_to_core23_tensor(HugeCTR::Tensor2<T> native_tensor,
                                                core23::Device device) {
  HCTR_CHECK_HINT(native_tensor.allocated(),
                  "can't convert non-allocated Tensor2 to core23::Tensor");
  auto origin_dims = native_tensor.get_dimensions();
  std::vector<int64_t> original_dims_i64(origin_dims.begin(), origin_dims.end());
  core23::Shape new_shape(original_dims_i64);
  auto ret = core23::Tensor::bind(native_tensor.get_ptr(), new_shape,
                                  core23::ToScalarType<T>::value, device);
  return ret;
}
template <typename T>
core23::Tensor convert_tensorbag_to_core23_tensor(HugeCTR::TensorBag2 bag, core23::Device device) {
  Tensor2<T> native_tensor = Tensor2<T>::stretch_from(bag);
  return convert_tensor2_to_core23_tensor(native_tensor, device);
}

template <typename T>
core23::Tensor convert_sparse_tensor_to_core_tensor(HugeCTR::SparseTensor<T> sparse_tensor,
                                                    core23::Device device) {
  auto native_tensor = sparse_tensor.get_value_tensor();
  return core23::Tensor::bind(native_tensor.get_ptr(),
                              {static_cast<int64_t>(native_tensor.get_num_elements())},
                              {core23::ToScalarType<T>::value}, device);
}

template <typename T>
std::vector<core23::Tensor> convert_sparse_tensors_to_core23_tensors(
    std::vector<HugeCTR::SparseTensor<T>> sparse_tensors, core23::Device device) {
  std::vector<core23::Tensor> core_tensors;
  for (auto& t : sparse_tensors) {
    core_tensors.push_back(convert_sparse_tensor_to_core_tensor(t, device));
  }
  return core_tensors;
}
template <typename T>
SparseTensor<T> convert_sparse_tensor23_to_sparse_tensor(const HugeCTR::SparseTensor23& sparse23) {
  auto value_tensor = sparse23.get_value_tensor();
  auto off_tensor = sparse23.get_rowoffset_tensor();
  auto shape = value_tensor.shape();
  std::vector<size_t> val_dimensions(shape.data(), shape.data() + shape.dims());
  auto val_buffer = PreallocatedBuffer2<T>::create(value_tensor.data(), val_dimensions);

  shape = off_tensor.shape();
  std::vector<size_t> off_dimensions(shape.data(), shape.data() + shape.dims());
  auto off_buffer = PreallocatedBuffer2<T>::create(off_tensor.data(), off_dimensions);
  SparseTensor<T> sparse_tensor(val_dimensions, val_buffer, off_buffer, sparse23.get_nnz_ptr(),
                                off_tensor.num_elements());
}
template <typename T>
SparseTensors<T> convert_sparse_tensors23_to_sparse_tensors(
    const std::vector<HugeCTR::SparseTensor23>& sparse23s) {
  SparseTensors<T> ret;
  for (const auto& sparse23 : sparse23s) {
    auto value_tensor = sparse23.get_value_tensor();
    auto off_tensor = sparse23.get_rowoffset_tensor();
    auto shape = value_tensor.shape();
    std::vector<size_t> val_dimensions(shape.data(), shape.data() + shape.dims());
    auto val_buffer = PreallocatedBuffer2<T>::create(value_tensor.data(), val_dimensions);

    shape = off_tensor.shape();
    std::vector<size_t> off_dimensions(shape.data(), shape.data() + shape.dims());
    auto off_buffer = PreallocatedBuffer2<T>::create(off_tensor.data(), off_dimensions);
    SparseTensor<T> sparse_tensor(val_dimensions, val_buffer, off_buffer, sparse23.get_nnz_ptr(),
                                  off_tensor.num_elements());
    ret.push_back(sparse_tensor);
  }
  return ret;
}

}  // namespace core_helper

}  // namespace HugeCTR