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

#include <cuda_runtime.h>

#include <core/core.hpp>
#include <resource_manager.hpp>
#include <unordered_map>

namespace hctr_internal {

using HugeCTR::CudaDeviceContext;

class GPUResource final : public core::GPUResourceBase {
  std::shared_ptr<HugeCTR::GPUResource> gpu_resource_;

 public:
  HCTR_DISALLOW_COPY_AND_MOVE(GPUResource);

  GPUResource(std::shared_ptr<HugeCTR::GPUResource> gpu_resource)
      : gpu_resource_(std::move(gpu_resource)) {}

  void set_stream(const std::string &name) override { gpu_resource_->set_stream(name); }

  std::string get_current_stream_name() override {
    return gpu_resource_->get_current_stream_name();
  }

  cudaStream_t get_stream() override { return gpu_resource_->get_stream(); }
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
        gpu_resource_(std::make_shared<GPUResource>(ext_->get_local_gpu(local_id))) {}

  std::shared_ptr<core::GPUResourceBase> get_local_gpu() override { return gpu_resource_; }

  const ncclComm_t &get_nccl() const override { return ext_->get_local_gpu(local_id_)->get_nccl(); }

  int get_local_gpu_id() const override { return local_id_; }

  int get_global_gpu_id() const override { return global_id_; }

  int get_device_id() const override { return device_id_; }

  size_t get_local_gpu_count() const override { return ext_->get_local_gpu_count(); }

  size_t get_global_gpu_count() const override { return ext_->get_global_gpu_count(); }

  int get_gpu_global_id_from_local_id(int local_id) const override {
    return ext_->get_gpu_global_id_from_local_id(local_id);
  }

  int get_gpu_local_id_from_global_id(int global_id) const override {
    return ext_->get_gpu_local_id_from_global_id(global_id);
  }
};

}  // namespace hctr_internal