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

// clang-format off
#include <cuda_runtime.h>

#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"

#if TF_VERSION_MAJOR == 1
#include "lookup/impl/core_impl/compat/gpu_process_state.h"
#else
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif

#include "HugeCTR/core23/device.hpp"
#include "HugeCTR/core23/allocator.hpp"
#include "HugeCTR/core23/allocator_params.hpp"
#include "HugeCTR/core23/cuda_stream.hpp"

namespace tf_internal {

namespace core23=HugeCTR::core23;
using tensorflow::AllocationAttributes;
using tensorflow::Allocator;
using tensorflow::AllocatorAttributes;
using tensorflow::DataType;
using tensorflow::DT_UINT8;
using tensorflow::GPUOptions;
using tensorflow::GPUProcessState;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::ResourceExhausted;
using tensorflow::GpuIdManager;

#if TF_VERSION_MAJOR == 1
using tensorflow::PlatformGpuId;
using tensorflow::TfGpuId;
#else
#ifndef TF_GE_211
using tensorflow::PlatformDeviceId;
using tensorflow::TfDeviceId;
#else
using tsl::PlatformDeviceId;
using tsl::TfDeviceId;
#endif
#endif

// We suppose one allocation can not allocate more than 1 TB
constexpr size_t MAX_MEMORY_SIZE = 1024 * 1024 * 1024 * 1024lu;

class TFAllocatorImpl  : public core23::Allocator{
 public:

  TFAllocatorImpl(const core23::Device &device){
    gpu_option_.set_allow_growth(true);
    if (device.type() == core23::DeviceType::GPU) {
      set_on_gpu(true);
      set_gpu_id(device.index());
    } else if (device.type() == core23::DeviceType::CPU) {
      set_on_gpu(false);
      set_cpu_id(0);
    } else {
      LOG(FATAL) << "Invalid Device type: " << device.type();
    }

    if (on_gpu_) {
      allocator_ =
#if TF_VERSION_MAJOR == 1
          GPUProcessState::singleton()->GetGPUAllocator(gpu_option_, gpu_id_, MAX_MEMORY_SIZE);
#else
          GPUProcessState::singleton()->GetGPUAllocator(gpu_option_, gpu_id_, MAX_MEMORY_SIZE, {});
#endif

    } else {
      // CPU Id should be 0 at almost time.
      allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(cpu_id_);
    }

  }

  void* allocate(int64_t size, core23::CUDAStream) override {
    void* ptr = allocator_->AllocateRaw(default_alignment(),size);
    if (ptr==nullptr) {
       LOG(FATAL) << "OOM when allocating SOK buffer";
    }
    return ptr;
  }

  void deallocate(void* ptr, core23::CUDAStream) override{

    if(ptr==nullptr){
      LOG(FATAL) << "SOK buffer be deallocated should't be nullptr";
    }
    allocator_->DeallocateRaw(ptr);
  }

  int64_t default_alignment() const { return alignof(std::max_align_t); }

  void set_on_gpu(bool on_gpu) { on_gpu_ = on_gpu; }

  void set_cpu_id(int cpu_id) { cpu_id_ = cpu_id; }
  void set_gpu_id(int gpu_id) {
#if TF_VERSION_MAJOR == 1
    PlatformGpuId platform_id;
    int tf_id = 0;
    while (GpuIdManager::TfToPlatformGpuId(TfGpuId(tf_id), &platform_id).ok()) {
      if (platform_id.value() == gpu_id) {
        gpu_id_ = TfGpuId(tf_id);
        return;
      }
      tf_id++;
    }
#else
    PlatformDeviceId platform_id;
    int tf_id = 0;
    while (GpuIdManager::TfToPlatformDeviceId(TfDeviceId(tf_id), &platform_id).ok()) {
      if (platform_id.value() == gpu_id) {
        gpu_id_ = TfDeviceId(tf_id);
        return;
      }
      tf_id++;
    }
#endif
    LOG(FATAL) << "Set TfDeviceId failed!";
  }

 private:
  bool on_gpu_;

  // Without using `GPUOptions::visible_device_list` in TensorFlow,
  // the device IDs of HCTR are equal to the platform GPU IDs and the
  // TF GPU IDs of TensorFlow. Please refer to here:
  // https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/common_runtime/device/device_id.h#L70/
#if TF_VERSION_MAJOR == 1
  TfGpuId gpu_id_;
#else
  TfDeviceId gpu_id_;
#endif

  int cpu_id_;         // refer to tensorflow::GPUProcessState::GetGpuHostAllocator
  tensorflow::Allocator *allocator_;
  GPUOptions gpu_option_;
};

void set_default_alloctor(){  
   core23::AllocatorParams::default_allocator_factory= [](const auto& params, const auto& device) {
    return std::unique_ptr<core23::Allocator>(new TFAllocatorImpl(device));
  };
}

}  // namespace tf_internal
