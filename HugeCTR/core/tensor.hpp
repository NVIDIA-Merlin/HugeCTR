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
#include <memory>

#include "HugeCTR/include/utils.hpp"
#include "core.hpp"
#include "datatype.hpp"
#include "device.hpp"
#include "shape.hpp"

namespace core {
using HugeCTR::CudaDeviceContext;
class CoreResourceManager;

struct TensorMeta final {
  Shape shape_;
  Device device_;
  DataType dtype_;

  TensorMeta(const Shape &shape, Device device, DataType dtype)
      : shape_(shape), device_(device), dtype_(dtype) {}
};

class TensorImpl {
  Storage storage_;
  int64_t storage_offset_;
  TensorMeta tensor_meta_;

 public:
  TensorImpl(Storage storage, int64_t storage_offset, const Shape &shape, Device device,
             DataType dtype)
      : storage_(storage), storage_offset_(storage_offset), tensor_meta_(shape, device, dtype) {}

  DISALLOW_COPY_AND_MOVE(TensorImpl)

  // void *get_ptr() { return reinterpret_cast<char *>(storage_->get_ptr()) + storage_offset_; }
  void *get_ptr() const {
    return static_cast<void *>(static_cast<char *>(storage_->get_ptr()) + storage_offset_);
  }

  void set_storage_offset(int64_t new_storage_offset) { storage_offset_ = new_storage_offset; }

  const TensorMeta &meta() const { return tensor_meta_; }

  const Shape &get_dimensions() const { return tensor_meta_.shape_; }

  const Device &device() const { return tensor_meta_.device_; }

  const DataType &dtype() const { return tensor_meta_.dtype_; }
};

void copy_bytes(size_t nbytes, const void *src, Device src_device, void *dst, Device dst_device,
                std::optional<cudaStream_t> stream = std::nullopt);

// we need this wrapper because if we directly use std::shared_ptr<ITensor>, it's hard to obtain
// right const semantic on get().
class Tensor final {
  std::shared_ptr<TensorImpl> tensor_;

  void check_initialized() const {
    HCTR_CHECK_HINT(tensor_ != nullptr, "Tensor is not initialized.");
  }

  template <typename T>
  void check_dtype() const {
    HCTR_CHECK_HINT(dtype().match<T>(), "T is not match with Tensor dtype");
  }

 public:
  Tensor() : tensor_(nullptr) {}

  Tensor(std::shared_ptr<TensorImpl> tensor) : tensor_(tensor) {}

  void *get() const {
    check_initialized();
    return tensor_->get_ptr();
  }

  template <typename T>
  T *get() {
    check_initialized();
    check_dtype<T>();
    return reinterpret_cast<T *>(tensor_->get_ptr());
  }

  template <typename T>
  const T *get() const {
    check_initialized();
    check_dtype<T>();
    return reinterpret_cast<const T *>(tensor_->get_ptr());
  }

  const Shape &get_dimensions() const {
    check_initialized();
    return tensor_->meta().shape_;
  }

  DataType dtype() const {
    check_initialized();
    return tensor_->meta().dtype_;
  }

  Device device() const {
    check_initialized();
    return tensor_->meta().device_;
  }

  int64_t get_num_elements() const { return get_dimensions().num_elements(); }

  size_t nbytes() const { return dtype().itemsize() * get_num_elements(); }

  template <typename T>
  void copy_from(const std::vector<T> &other, std::optional<cudaStream_t> stream = std::nullopt) {
    HCTR_CHECK_HINT(match<T>(), "Tensor.from() type not match ");
    HCTR_CHECK_HINT(nbytes() >= sizeof(T) * other.size(),
                    "Tensor.from() dst nbytes %lu should >= src nbytes() %lu", nbytes(),
                    sizeof(T) * other.size());

    copy_bytes(sizeof(T) * other.size(), reinterpret_cast<const void *>(other.data()),
               DeviceType::CPU, get(), device(), stream);
  }

  void copy_from(const Tensor &other, std::optional<cudaStream_t> stream = std::nullopt);

  template <typename T>
  void copy_to(std::vector<T> &other, std::optional<cudaStream_t> stream = std::nullopt) const {
    HCTR_CHECK_HINT(match<T>(), "Tensor.from() type not match ");
    HCTR_CHECK_HINT(sizeof(T) * other.size() >= nbytes(),
                    "Tensor.to() dst nbytes %lu should >= src nbytes() %lu",
                    sizeof(T) * other.size(), nbytes());
    size_t bytes_copy = nbytes();

    copy_bytes(bytes_copy, get(), device(), reinterpret_cast<void *>(other.data()), DeviceType::CPU,
               stream);
  }

  void copy_to(Tensor &other, std::optional<cudaStream_t> stream = std::nullopt) const;

  Tensor to(std::shared_ptr<CoreResourceManager> core, Device device,
            std::optional<cudaStream_t> stream = std::nullopt) const;

  template <typename T>
  void to(std::vector<T> *other, std::optional<cudaStream_t> stream = std::nullopt) const {
    other->clear();
    HCTR_CHECK_HINT(match<T>(), "Tensor.to() type not match ");

    other->resize(get_num_elements());
    copy_bytes(nbytes(), get(), device(), other->data(), DeviceType::CPU, stream);
  }

  void zeros(std::optional<cudaStream_t> stream = std::nullopt);

  // Check function
  template <typename T>
  bool match() const {
    return dtype().match<T>();
  }
};

// Currenly this class trys to own the underlying data, because we use this class in the ILookup as
// the return result from lookup(), so this class should make sure the the data will not leak after
// we do lookup() and this complicates the implementation.
//
// It will be easier for us to make this class as a *device side* const reference to
// std::vector<Tensor>, so all we need from this class is get() and ctor. The user decide that the
// lifetime of the data this class reference to is longer than the lifetime of this class. We trys
// to own the underlying data  currently
class TensorList final {
  std::shared_ptr<TensorImpl> device_tensor_list_;
  std::vector<Tensor> host_tensor_list_;
  bool can_visit_from_host_;

  void check_initialized() const {
    HCTR_CHECK_HINT(device_tensor_list_ != nullptr, "TensorList is not initialized.");
  }

  template <typename T, typename = typename std::enable_if_t<is_integer<T>>>
  TensorList(CoreResourceManager *backend, T size, Device device, DataType dtype,
             bool can_visit_from_host)
      : can_visit_from_host_(can_visit_from_host) {
    Storage storage = backend->CreateStorage(device);
    storage->extend(static_cast<size_t>(size) * sizeof(void *));
    storage->allocate();
    device_tensor_list_ = std::make_shared<TensorImpl>(storage, 0, size, device, dtype);
  }

 public:
  TensorList() : device_tensor_list_(nullptr) {}

  template <typename T, typename = typename std::enable_if_t<is_integer<T>>>
  TensorList(CoreResourceManager *backend, T size, Device device, DataType dtype)
      : TensorList(backend, size, device, dtype, false) {}

  TensorList(CoreResourceManager *backend, const std::vector<Tensor> &tensor_list, Device device,
             DataType dtype, std::optional<cudaStream_t> stream = std::nullopt)
      : TensorList(backend, tensor_list.size(), device, dtype, true) {
    CudaDeviceContext ctx(device.index());
    HCTR_CHECK_HINT(std::all_of(tensor_list.begin(), tensor_list.end(),
                                [&](const Tensor &t) { return t.dtype() == dtype; }),
                    "TensorList: dtype is not the same.");

    std::vector<const void *> pointers;
    std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(pointers),
                   [](const Tensor &t) { return t.get(); });
    int64_t nbytes = pointers.size() * sizeof(void *);
    if (stream.has_value()) {
      HCTR_LIB_THROW(cudaMemcpyAsync(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                     cudaMemcpyHostToDevice, stream.value()));
    } else {
      HCTR_LIB_THROW(cudaMemcpy(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                cudaMemcpyHostToDevice));
    }

    for (size_t i = 0; i < tensor_list.size(); ++i) {
      host_tensor_list_.push_back(tensor_list[i]);
    }
  }

  TensorList(CoreResourceManager *backend, const Tensor &tensor, const std::vector<int64_t> &offset,
             const std::vector<int64_t> &length, Device device, DataType dtype,
             std::optional<cudaStream_t> stream = std::nullopt)
      : TensorList(backend, offset.size(), device, dtype, false) {
    CudaDeviceContext ctx(device.index());
    HCTR_CHECK_HINT(tensor.dtype() == dtype,
                    "TensorList: attempted to initialize with different dtype");

    std::vector<const void *> pointers;
    for (size_t i = 0; i < offset.size(); ++i) {
      pointers.push_back(reinterpret_cast<const char *>(tensor.get()) +
                         offset[i] * tensor.dtype().itemsize());
    }

    int64_t nbytes = offset.size() * sizeof(void *);
    if (stream.has_value()) {
      HCTR_LIB_THROW(cudaMemcpyAsync(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                     cudaMemcpyHostToDevice, stream.value()));
    } else {
      HCTR_LIB_THROW(cudaMemcpy(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                cudaMemcpyHostToDevice));
    }

    host_tensor_list_.push_back(tensor);
  }

  const Tensor &operator[](size_t idx) const {
    HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid operator[]");
    HCTR_CHECK_HINT(idx < host_tensor_list_.size(),
                    "TensorList: invalid index. Idx = %lu; Size = %lu", idx,
                    host_tensor_list_.size());
    return host_tensor_list_[idx];
  }

  const Tensor &at(size_t idx) const {
    HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid at()");
    HCTR_CHECK_HINT(idx < host_tensor_list_.size(),
                    "TensorList: invalid index. Idx = %lu; Size = %lu", idx,
                    host_tensor_list_.size());
    return host_tensor_list_[idx];
  }

  const std::vector<Tensor> &host_tensor_list() const {
    HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid host_tensor_list");
    return host_tensor_list_;
  }

  void *get() const {
    check_initialized();
    return device_tensor_list_->get_ptr();
  }

  template <typename T>
  T **get() {
    check_initialized();
    HCTR_CHECK_HINT(dtype().match<T>(),
                    "TensorList.get<T>() error. T is not match with TensorList dtype");
    return reinterpret_cast<T **>(device_tensor_list_->get_ptr());
  }

  template <typename T>
  const T **get() const {
    check_initialized();
    HCTR_CHECK_HINT(dtype().match<T>(),
                    "TensorList.get<T>() error. T is not match with TensorList dtype");
    return reinterpret_cast<const T **>(device_tensor_list_->get_ptr());
  }

  DataType dtype() const {
    check_initialized();
    return device_tensor_list_->meta().dtype_;
  }

  Device device() const {
    check_initialized();
    return device_tensor_list_->meta().device_;
  }

  int64_t get_num_elements() const {
    check_initialized();
    return device_tensor_list_->meta().shape_.num_elements();
  }

  size_t nbytes() const { return dtype().itemsize() * get_num_elements(); }
};
}  // namespace core
