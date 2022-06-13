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
#include "tensor.hpp"

#include <iostream>

#include "core.hpp"
#include "macro.hpp"
#include "resource_manager.hpp"
#include "buffer.hpp"

namespace core {

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
                std::optional<cudaStream_t> stream) {
  auto copy_bytes_impl =
      copy_bytes_functions[stream.has_value() ? 1 : 0][static_cast<int>(src_device.type())]
                          [static_cast<int>(dst_device.type())];
  copy_bytes_impl(nbytes, src, dst, stream);
}

// TODO: src CPUGPU
void Tensor::copy_from(const Tensor &other, std::optional<cudaStream_t> stream) {
  HCTR_CHECK_HINT(this->nbytes() >= other.nbytes(),
                  "copy from tensor dst bytes %lu should >= src bytes %lu", this->nbytes(),
                  other.nbytes());
  HCTR_CHECK_HINT(this->dtype() == other.dtype(), "copy tensor dtype not match");

  copy_bytes(other.nbytes(), other.get(), other.device(), get(), device(), stream);
}

void Tensor::copy_to(Tensor &other, std::optional<cudaStream_t> stream) const {
  HCTR_CHECK_HINT(this->nbytes() == other.nbytes(), "copy to tensor src bytes %lu <= dst bytes %lu",
                  this->nbytes(), other.nbytes());
  HCTR_CHECK_HINT(this->dtype() == other.dtype(), "copy tensor dtype not match");

  copy_bytes(nbytes(), get(), device(), other.get(), other.device(), stream);
}

Tensor Tensor::to(std::shared_ptr<CoreResourceManager> core, Device device, std::optional<cudaStream_t> stream) const {
  BufferPtr buffer_ptr = GetBuffer(core);
  Tensor t = buffer_ptr->reserve(get_dimensions(), device, dtype());
  buffer_ptr->allocate();
  copy_to(t, stream);
  return t;
}

void Tensor::zeros(std::optional<cudaStream_t> stream) {
  auto device = tensor_->device();
  if (device.is_gpu()) {
    CudaDeviceContext context(device.index());
    if (stream.has_value()) {
      HCTR_LIB_THROW(cudaMemsetAsync(tensor_->get_ptr(), 0, nbytes(), stream.value()));
    } else {
      HCTR_LIB_THROW(cudaMemset(tensor_->get_ptr(), 0, nbytes()));
    }
  } else {
    memset(tensor_->get_ptr(), 0, nbytes());
  }
}

}  // namespace core