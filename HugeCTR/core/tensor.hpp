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
/*
This header file will be included in sok and be compiled with tensorflow by c++14.
So make sure this headerfile is c++14 compatible.
We use cudaStream_t stream = 0 to represent the use case that user do not specify cudaStream_t in
copy function. signature
*/
#pragma once
#include <nccl.h>

#include <memory>

#include "core.hpp"
#include "datatype.hpp"
#include "device.hpp"
#include "shape.hpp"

namespace core {
class CoreResourceManager;

ncclDataType_t get_nccl_dtype_from_tensor_scalar_type(TensorScalarType scalar_type);

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

  void *get_ptr() const {
    return static_cast<void *>(static_cast<char *>(storage_->get_ptr()) + storage_offset_);
  }

  void set_storage_offset(int64_t new_storage_offset) { storage_offset_ = new_storage_offset; }

  const TensorMeta &meta() const { return tensor_meta_; }

  const Shape &get_dimensions() const { return tensor_meta_.shape_; }

  const Device &device() const { return tensor_meta_.device_; }

  const DataType &dtype() const { return tensor_meta_.dtype_; }
};

// we need this wrapper because if we directly use std::shared_ptr<ITensor>, it's hard to obtain
// right const semantic on get().
class Tensor final {
  std::shared_ptr<TensorImpl> tensor_;

  void check_initialized() const;

  template <typename T>
  void check_dtype() const;

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
  void copy_from(const std::vector<T> &other, cudaStream_t stream = 0);

  void copy_from(const Tensor &other, cudaStream_t stream = 0);

  template <typename T>
  void copy_to(std::vector<T> &other, cudaStream_t stream = 0) const;

  void copy_to(Tensor &other, cudaStream_t stream = 0) const;

  Tensor to(std::shared_ptr<CoreResourceManager> core, Device device,
            cudaStream_t stream = 0) const;

  template <typename T>
  void to(std::vector<T> *other, cudaStream_t stream = 0) const;

  template <typename T>
  inline std::vector<T> to_vector(cudaStream_t stream = 0) const {
    std::vector<T> v;
    to(&v, stream);
    return v;
  }

  void zeros(cudaStream_t stream = 0);

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

  void check_initialized() const;

#if __cplusplus >= 201703L
  template <typename T, typename = typename std::enable_if_t<is_integer<T>>>
  TensorList(CoreResourceManager *backend, T size, Device device, DataType dtype,
             bool can_visit_from_host)
      : can_visit_from_host_(can_visit_from_host) {
    Storage storage = backend->CreateStorage(device);
    storage->extend(static_cast<size_t>(size) * sizeof(void *));
    storage->allocate();
    device_tensor_list_ = std::make_shared<TensorImpl>(storage, 0, size, device, dtype);
  }
#endif

 public:
  TensorList() : device_tensor_list_(nullptr) {}

#if __cplusplus >= 201703L
  template <typename T, typename = typename std::enable_if_t<is_integer<T>>>
  TensorList(CoreResourceManager *backend, T size, Device device, DataType dtype)
      : TensorList(backend, size, device, dtype, false) {}
#endif

  TensorList(CoreResourceManager *backend, const std::vector<Tensor> &tensor_list, Device device,
             DataType dtype, cudaStream_t stream = 0);

  TensorList(CoreResourceManager *backend, const Tensor &tensor, const std::vector<int64_t> &offset,
             const std::vector<int64_t> &length, Device device, DataType dtype,
             cudaStream_t stream = 0);

  const Tensor &operator[](size_t idx) const;

  const Tensor &at(size_t idx) const;

  const std::vector<Tensor> &host_tensor_list() const;

  void *get() const {
    check_initialized();
    return device_tensor_list_->get_ptr();
  }

  template <typename T>
  T **get();

  template <typename T>
  const T **get() const;

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
