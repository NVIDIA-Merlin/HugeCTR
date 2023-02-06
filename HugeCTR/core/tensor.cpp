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

#include <core/buffer.hpp>
#include <core/core.hpp>
#include <core/macro.hpp>
#include <core/tensor.hpp>
#include <iostream>
#include <utils.hpp>

namespace core {
using HugeCTR::CudaDeviceContext;

ncclDataType_t get_nccl_dtype_from_tensor_scalar_type(TensorScalarType scalar_type) {
  switch (scalar_type) {
    case TensorScalarType::Void:
      return ncclChar;
    case TensorScalarType::Float32:
      return ncclFloat32;
    case TensorScalarType::Float16:
      return ncclHalf;
    case TensorScalarType::Int64:
      return ncclInt64;
    case TensorScalarType::UInt64:
      return ncclUint64;
    case TensorScalarType::Int32:
      return ncclInt32;
    case TensorScalarType::UInt32:
      return ncclUint32;
    case TensorScalarType::Size_t:
      return ncclInt64;
    case TensorScalarType::Char:
      return ncclChar;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                     "Not supported TensorScalarType to NcclDataType_t");
  }
  return ncclInt;
}

void Tensor::check_initialized() const {
  HCTR_CHECK_HINT(tensor_ != nullptr, "Tensor is not initialized.");
}

template <typename T>
void Tensor::check_dtype() const {
  HCTR_CHECK_HINT(dtype().match<T>(), "T is not match with Tensor dtype");
}

using CopyBytesFunction = void (*)(size_t bytes_size, const void *src, void *dst,
                                   std::optional<cudaStream_t> stream);

void copy_bytes_host(size_t bytes_size, const void *src, void *dst,
                     std::optional<cudaStream_t> stream) {
  memcpy(dst, src, bytes_size);
}

void copy_bytes_h2d(size_t bytes_size, const void *src, void *dst,
                    std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpy(dst, src, bytes_size, cudaMemcpyHostToDevice));
}

void copy_bytes_h2d_async(size_t bytes_size, const void *src, void *dst,
                          std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpyAsync(dst, src, bytes_size, cudaMemcpyHostToDevice, stream.value()));
}

void copy_bytes_d2h(size_t bytes_size, const void *src, void *dst,
                    std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpy(dst, src, bytes_size, cudaMemcpyDeviceToHost));
}

void copy_bytes_d2h_async(size_t bytes_size, const void *src, void *dst,
                          std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpyAsync(dst, src, bytes_size, cudaMemcpyDeviceToHost, stream.value()));
}

void copy_bytes_d2d(size_t bytes_size, const void *src, void *dst,
                    std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpy(dst, src, bytes_size, cudaMemcpyDeviceToDevice));
}

void copy_bytes_d2d_async(size_t bytes_size, const void *src, void *dst,
                          std::optional<cudaStream_t> stream) {
  HCTR_LIB_THROW(cudaMemcpyAsync(dst, src, bytes_size, cudaMemcpyDeviceToDevice, stream.value()));
}

static CopyBytesFunction copy_bytes_functions[2][static_cast<int>(DeviceType::MAX_DEVICE_NUM)]
                                             [static_cast<int>(DeviceType::MAX_DEVICE_NUM)];

struct _CopyBytesFunctionRegister {
  _CopyBytesFunctionRegister(DeviceType from_type, DeviceType to_type, CopyBytesFunction copy_sync,
                             CopyBytesFunction copy_async) {
    auto from = static_cast<int>(from_type);
    auto to = static_cast<int>(to_type);

    HCTR_CHECK_HINT(copy_bytes_functions[0][from][to] == nullptr &&
                        copy_bytes_functions[1][from][to] == nullptr,
                    "duplicate copy bytes functions register");
    copy_bytes_functions[0][from][to] = copy_sync;
    copy_bytes_functions[1][from][to] = copy_async;
  }
};

#define REGISTER_COPY_BYTES_FUNCTION(from, to, copy_sync, copy_async)                  \
  static _CopyBytesFunctionRegister ANONYMOUS_VARIABLE(copy_bytes_function_reggister)( \
      from, to, copy_sync, copy_async);

// src host
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU, DeviceType::CPU, copy_bytes_host, copy_bytes_host);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU, DeviceType::GPU, copy_bytes_h2d,
                             copy_bytes_h2d_async);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU, DeviceType::CPUGPU, copy_bytes_h2d,
                             copy_bytes_h2d_async);

// src cuda
REGISTER_COPY_BYTES_FUNCTION(DeviceType::GPU, DeviceType::CPU, copy_bytes_d2h,
                             copy_bytes_d2h_async);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::GPU, DeviceType::GPU, copy_bytes_d2d,
                             copy_bytes_d2d_async);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::GPU, DeviceType::CPUGPU, copy_bytes_d2d,
                             copy_bytes_d2d_async);

// src cpucuda
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPUGPU, DeviceType::CPU, copy_bytes_d2h,
                             copy_bytes_d2h_async);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPUGPU, DeviceType::GPU, copy_bytes_d2d,
                             copy_bytes_d2d_async);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPUGPU, DeviceType::CPUGPU, copy_bytes_d2d,
                             copy_bytes_d2d_async);

void copy_bytes(size_t nbytes, const void *src, Device src_device, void *dst, Device dst_device,
                cudaStream_t stream) {
  auto copy_bytes_impl =
      copy_bytes_functions[(stream != 0) ? 1 : 0][static_cast<int>(src_device.type())]
                          [static_cast<int>(dst_device.type())];
  copy_bytes_impl(nbytes, src, dst, stream);
}

template <typename T>
void Tensor::copy_from(const std::vector<T> &other, cudaStream_t stream) {
  HCTR_CHECK_HINT(match<T>(), "Tensor.from() type not match ");
  HCTR_CHECK_HINT(nbytes() >= sizeof(T) * other.size(),
                  "Tensor.from() dst nbytes %lu should >= src nbytes() %lu", nbytes(),
                  sizeof(T) * other.size());

  copy_bytes(sizeof(T) * other.size(), reinterpret_cast<const void *>(other.data()),
             DeviceType::CPU, get(), device(), stream);
}

// TODO: src CPUGPU
void Tensor::copy_from(const Tensor &other, cudaStream_t stream) {
  HCTR_CHECK_HINT(this->nbytes() >= other.nbytes(),
                  "copy from tensor dst bytes %lu should >= src bytes %lu", this->nbytes(),
                  other.nbytes());
  HCTR_CHECK_HINT(this->dtype() == other.dtype(), "copy tensor dtype not match");

  copy_bytes(other.nbytes(), other.get(), other.device(), get(), device(), stream);
}

template <typename T>
void Tensor::copy_to(std::vector<T> &other, cudaStream_t stream) const {
  HCTR_CHECK_HINT(match<T>(), "Tensor.from() type not match ");
  HCTR_CHECK_HINT(sizeof(T) * other.size() >= nbytes(),
                  "Tensor.to() dst nbytes %lu should >= src nbytes() %lu", sizeof(T) * other.size(),
                  nbytes());
  size_t bytes_copy = nbytes();

  copy_bytes(bytes_copy, get(), device(), reinterpret_cast<void *>(other.data()), DeviceType::CPU,
             stream);
}

void Tensor::copy_to(Tensor &other, cudaStream_t stream) const {
  HCTR_CHECK_HINT(this->nbytes() == other.nbytes(), "copy to tensor src bytes %lu <= dst bytes %lu",
                  this->nbytes(), other.nbytes());
  HCTR_CHECK_HINT(this->dtype() == other.dtype(), "copy tensor dtype not match");

  copy_bytes(nbytes(), get(), device(), other.get(), other.device(), stream);
}

template <typename T>
void Tensor::to(std::vector<T> *other, cudaStream_t stream) const {
  other->clear();
  HCTR_CHECK_HINT(match<T>(), "Tensor.to() type not match ");

  other->resize(get_num_elements());
  copy_bytes(nbytes(), get(), device(), other->data(), DeviceType::CPU, stream);
}

Tensor Tensor::to(std::shared_ptr<CoreResourceManager> core, Device device,
                  cudaStream_t stream) const {
  BufferPtr buffer_ptr = GetBuffer(core);
  Tensor t = buffer_ptr->reserve(get_dimensions(), device, dtype());
  buffer_ptr->allocate();
  copy_to(t, stream);
  return t;
}

void Tensor::zeros(cudaStream_t stream) {
  auto device = tensor_->device();
  if (device.is_gpu()) {
    CudaDeviceContext context(device.index());
    if (stream != 0) {
      HCTR_LIB_THROW(cudaMemsetAsync(tensor_->get_ptr(), 0, nbytes(), stream));
    } else {
      HCTR_LIB_THROW(cudaMemset(tensor_->get_ptr(), 0, nbytes()));
    }
  } else {
    memset(tensor_->get_ptr(), 0, nbytes());
  }
}

void TensorList::check_initialized() const {
  HCTR_CHECK_HINT(device_tensor_list_ != nullptr, "TensorList is not initialized.");
}

TensorList::TensorList(CoreResourceManager *backend, const std::vector<Tensor> &tensor_list,
                       Device device, DataType dtype, cudaStream_t stream)
    : TensorList(backend, tensor_list.size(), device, dtype, true) {
  CudaDeviceContext ctx(device.index());
  HCTR_CHECK_HINT(std::all_of(tensor_list.begin(), tensor_list.end(),
                              [&](const Tensor &t) { return t.dtype() == dtype; }),
                  "TensorList: dtype is not the same.");

  std::vector<const void *> pointers;
  std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(pointers),
                 [](const Tensor &t) { return t.get(); });
  int64_t nbytes = pointers.size() * sizeof(void *);
  if (stream != 0) {
    HCTR_LIB_THROW(cudaMemcpyAsync(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                   cudaMemcpyHostToDevice, stream));
  } else {
    HCTR_LIB_THROW(cudaMemcpy(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                              cudaMemcpyHostToDevice));
  }

  for (size_t i = 0; i < tensor_list.size(); ++i) {
    host_tensor_list_.push_back(tensor_list[i]);
  }
}

TensorList::TensorList(CoreResourceManager *backend, const Tensor &tensor,
                       const std::vector<int64_t> &offset, const std::vector<int64_t> &length,
                       Device device, DataType dtype, cudaStream_t stream)
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
  if (stream != 0) {
    HCTR_LIB_THROW(cudaMemcpyAsync(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                                   cudaMemcpyHostToDevice, stream));
  } else {
    HCTR_LIB_THROW(cudaMemcpy(device_tensor_list_->get_ptr(), pointers.data(), nbytes,
                              cudaMemcpyHostToDevice));
  }

  host_tensor_list_.push_back(tensor);
}

const Tensor &TensorList::operator[](size_t idx) const {
  HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid operator[]");
  HCTR_CHECK_HINT(idx < host_tensor_list_.size(),
                  "TensorList: invalid index. Idx = %lu; Size = %lu", idx,
                  host_tensor_list_.size());
  return host_tensor_list_[idx];
}

const Tensor &TensorList::at(size_t idx) const {
  HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid at()");
  HCTR_CHECK_HINT(idx < host_tensor_list_.size(),
                  "TensorList: invalid index. Idx = %lu; Size = %lu", idx,
                  host_tensor_list_.size());
  return host_tensor_list_[idx];
}

const std::vector<Tensor> &TensorList::host_tensor_list() const {
  HCTR_CHECK_HINT(can_visit_from_host_, "TensorList: invalid host_tensor_list");
  return host_tensor_list_;
}

template <typename T>
T **TensorList::get() {
  check_initialized();
  HCTR_CHECK_HINT(dtype().match<T>(),
                  "TensorList.get<T>() error. T is not match with TensorList dtype");
  return reinterpret_cast<T **>(device_tensor_list_->get_ptr());
}

template <typename T>
const T **TensorList::get() const {
  check_initialized();
  HCTR_CHECK_HINT(dtype().match<T>(),
                  "TensorList.get<T>() error. T is not match with TensorList dtype");
  return reinterpret_cast<const T **>(device_tensor_list_->get_ptr());
}

// need to put those template func in cpp because we dont want to expose hugectr logging in header
// file
#define DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(type)                                   \
  template void Tensor::check_dtype<type>() const;                                            \
  template void Tensor::copy_from<type>(const std::vector<type> &other, cudaStream_t stream); \
  template void Tensor::copy_to<type>(std::vector<type> & other, cudaStream_t stream) const;  \
  template void Tensor::to<type>(std::vector<type> * other, cudaStream_t stream) const;       \
  template type **TensorList::get<type>();                                                    \
  template const type **TensorList::get<type>() const;

DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(float)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(__half)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(int64_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(long long)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(uint64_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(int32_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(uint32_t)
DEFINE_TEMPLATE_FUNC_IN_TENSOR_AND_TENSORLIST(char)

}  // namespace core