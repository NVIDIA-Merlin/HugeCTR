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

#include <memory>

#include <nccl.h>
#include "HugeCTR/include/tensor2.hpp"

namespace core {
class Device;
class BufferBlockImpl;
class BufferImpl;

using BufferBlockPtr = std::shared_ptr<BufferBlockImpl>;
using BufferPtr = std::shared_ptr<BufferImpl>;
using HugeCTR::TensorScalarType;

class IStorageImpl {
 public:
  virtual ~IStorageImpl() = default;

  virtual void *get_ptr() = 0;

  virtual size_t nbytes() const = 0;

  virtual void extend(size_t nbytes) = 0;

  virtual void allocate() = 0;
};

using Storage = std::shared_ptr<IStorageImpl>;

class RawPtrStorageImpl : public IStorageImpl {
  void *ptr_;
  size_t total_size_in_bytes_;

 public:
  RawPtrStorageImpl(void *ptr, size_t total_size_in_bytes) : ptr_(ptr), total_size_in_bytes_(total_size_in_bytes) {}

  DISALLOW_COPY_AND_MOVE(RawPtrStorageImpl)

  void *get_ptr() override {
    return ptr_;
  }

  size_t nbytes() const override { return total_size_in_bytes_; }

  void extend(size_t s) override {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "RawPtrStorageImpl can not extend");
  }

  void allocate() override {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "RawPtrStorageImpl can not allocate");
  }
};

class GPUResourceBase {
 public:
  virtual ~GPUResourceBase() = default;
  virtual void set_stream(const std::string &name) = 0;
  virtual std::string get_current_stream_name() = 0;
  virtual cudaStream_t get_stream() = 0;  // will return current stream
};

class StreamContext {
  std::string origin_stream_name_;
  std::shared_ptr<GPUResourceBase> local_gpu_;

 public:
  StreamContext(const std::shared_ptr<GPUResourceBase> &local_gpu,
                const std::string &new_stream_name)
      : origin_stream_name_(local_gpu->get_current_stream_name()) {
    local_gpu_->set_stream(new_stream_name);
  }
  ~StreamContext() { local_gpu_->set_stream(origin_stream_name_); }
};

class CoreResourceManager {
 public:
  virtual ~CoreResourceManager() = default;

  virtual std::shared_ptr<GPUResourceBase> get_local_gpu() = 0;

  virtual const ncclComm_t &get_nccl() const = 0;

  virtual Storage CreateStorage(Device device) = 0;

  virtual int get_local_gpu_id() const = 0;
  
  virtual int get_global_gpu_id() const = 0;
  
  virtual int get_device_id() const = 0;

  virtual size_t get_local_gpu_count() const = 0;

  virtual size_t get_global_gpu_count() const = 0;

  virtual int get_gpu_global_id_from_local_id(int local_id) const = 0;

  virtual int get_gpu_local_id_from_global_id(int global_id) const = 0;
};

inline ncclDataType_t get_nccl_dtype_from_tensor_scalar_type(TensorScalarType scalar_type) {
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
}  // namespace core