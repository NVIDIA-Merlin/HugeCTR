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

#include <unordered_map>

#include "../buffer.hpp"
#include "../core.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "allocator.hpp"

namespace hctr_internal {

using core::Device;
using core::DeviceType;
using HugeCTR::CudaDeviceContext;

// we suppose one allocation can not allocate more than 1 TB
constexpr size_t MAX_MEMORY_BUFFER_SIZE = 1024 * 1024 * 1024 * 1024lu;

class HCTRStorageImpl final : public core::IStorageImpl {
  void *ptr_;
  size_t total_size_in_bytes_;
  bool allocated_;
  Allocator *allocator_;

 public:
  HCTRStorageImpl(Allocator *allocator)
      : ptr_(nullptr), total_size_in_bytes_(0), allocated_(false), allocator_(allocator) {}

  DISALLOW_COPY_AND_MOVE(HCTRStorageImpl)

  ~HCTRStorageImpl() {
    if (allocated_) {
      allocator_->release(ptr_);
    }
  }

  void *get_ptr() override {
    HCTR_CHECK_HINT(allocated_, "Tensor is not allocated. You forget call allocate()?");
    return ptr_;
  }

  size_t nbytes() const override { return total_size_in_bytes_; }

  void extend(size_t s) override {
    HCTR_CHECK_HINT(s <= MAX_MEMORY_BUFFER_SIZE, "out of memory for reserving memory %lu", s);
    total_size_in_bytes_ += s;
  }

  void allocate() override {
    HCTR_CHECK_HINT(allocated_ == false, "InternalBuffer has been allocated!");
    HCTR_CHECK_HINT(total_size_in_bytes_ >= 0, "InternalBuffer size_in_bytes should >= 0");

    ptr_ = allocator_->allocate(total_size_in_bytes_);
    allocated_ = true;
  }
};

// used to wrap HugeCTR::Tensor2<T> into core::Tensor
class NativeHCTRStorageWrapper final : public core::IStorageImpl {
  void *ptr_;
  size_t total_size_in_bytes_;

 public:
  NativeHCTRStorageWrapper(void *ptr, size_t total_size_in_bytes)
      : ptr_(ptr), total_size_in_bytes_(total_size_in_bytes) {}

  DISALLOW_COPY_AND_MOVE(NativeHCTRStorageWrapper)

  void *get_ptr() override { return ptr_; }

  size_t nbytes() const override { return total_size_in_bytes_; }

  void extend(size_t s) override {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "NativeHCTRStorageWrapper can not extend");
  }

  void allocate() override {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "NativeHCTRStorageWrapper can not allocate");
  }
};

class GPUResource final : public core::GPUResourceBase {
  int device_id_;
  std::string current_stream_name_;
  std::unordered_map<std::string, cudaStream_t> stream_map_;

 public:
  GPUResource(int device_id, cudaStream_t default_stream)
      : device_id_(device_id), current_stream_name_("default") {
    stream_map_[current_stream_name_] = default_stream;
  }

  DISALLOW_COPY_AND_MOVE(GPUResource)

  void set_stream(const std::string &name) override { current_stream_name_ = name; }

  std::string get_current_stream_name() override { return current_stream_name_; }

  cudaStream_t get_stream() override {
    if (stream_map_.find(current_stream_name_) == stream_map_.end()) {
      CudaDeviceContext context(device_id_);
      cudaStream_t stream;
      HCTR_LIB_THROW(cudaStreamCreate(&stream));
      stream_map_[current_stream_name_] = stream;
    }
    return stream_map_.at(current_stream_name_);
  }
};

class HCTRCoreResourceManager : public core::CoreResourceManager {
  std::shared_ptr<HugeCTR::ResourceManager> ext_;
  int local_id_;
  int global_id_;
  int device_id_;

  std::shared_ptr<core::GPUResourceBase> gpu_resource_;

 public:
  HCTRCoreResourceManager(std::shared_ptr<HugeCTR::ResourceManager> ext, int local_id)
      : ext_(ext),
        local_id_(local_id),
        global_id_(ext_->get_gpu_global_id_from_local_id(local_id)),
        device_id_(ext_->get_local_gpu_device_id_list()[local_id]),
        gpu_resource_(std::make_shared<GPUResource>(device_id_,
                                                    ext_->get_local_gpu(local_id)->get_stream())) {}

  std::shared_ptr<core::GPUResourceBase> get_local_gpu() override { return gpu_resource_; }

  const ncclComm_t &get_nccl() const override { return ext_->get_local_gpu(local_id_)->get_nccl(); }

  core::Storage CreateStorage(Device device) {
    return std::make_shared<HCTRStorageImpl>(GetAllocator(device.type()));
  }

  int get_local_gpu_id() const override { return local_id_; }

  int get_global_gpu_id() const override { return global_id_; }

  int get_device_id() const override { return device_id_; }

  size_t get_local_gpu_count() const override { return ext_->get_local_gpu_count(); }

  size_t get_global_gpu_count() const override { return ext_->get_global_gpu_count(); }

  // int get_device_id_from_local_id(int local_id) const override {
  //   int device_id = ext_->get_local_gpu_device_id_list()[local_id];
  //   return device_id;
  // }

  int get_gpu_global_id_from_local_id(int local_id) const override {
    return ext_->get_gpu_global_id_from_local_id(local_id);
  }

  int get_gpu_local_id_from_global_id(int global_id) const override {
    return ext_->get_gpu_local_id_from_global_id(global_id);
  }
};
}  // namespace hctr_internal