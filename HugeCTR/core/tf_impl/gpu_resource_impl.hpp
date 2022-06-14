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

#include <assert.h>
#include <cuda_runtime.h>

#include <unordered_map>

#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tf_internal {

using core::GPUResourceBase;
using stream_executor::Stream;
using tensorflow::DeviceContext;
using tensorflow::GPUDeviceContext;
using tensorflow::OpKernelContext;

class GPUResource final : public GPUResourceBase {
 public:
  GPUResource(OpKernelContext *ctx) : ctx_(ctx), current_stream_name_("default") {
    DeviceContext *dc = ctx->op_device_context();
    if (!dc) {
      LOG(FATAL) << "Get DeviceContext fail! please check OpKernel running on GPU.";
    }
    const GPUDeviceContext *gpu_dc = static_cast<GPUDeviceContext *>(dc);
    cudaStream_t *stream =
        reinterpret_cast<cudaStream_t *>(gpu_dc->stream()->implementation()->GpuStreamMemberHack());

    // TODO(@hrong):we can also get the CUDA streams of NCCL, H2D, D2H and D2D
    // through APIs below.
    // se::Stream* nccl_stream();
    // se::Stream* host_to_device_stream();
    // se::Stream* device_to_host_stream();
    // se::Stream* device_to_device_stream(int index);
    if (!stream) {
      LOG(FATAL) << "Get default CUDA stream fail!";
    }
    stream_map_[current_stream_name_] = *stream;
  }

  DISALLOW_COPY_AND_MOVE(GPUResource)

  void set_stream(const std::string &name) override { current_stream_name_ = name; }

  std::string get_current_stream_name() override { return current_stream_name_; }

  cudaStream_t get_stream() override {
    if (stream_map_.find(current_stream_name_) == stream_map_.end()) {
      cudaStream_t stream;
      if (cudaStreamCreate(&stream) != cudaSuccess) {
        LOG(FATAL) << "Create CUDA stream fail!";
      }
      stream_map_[current_stream_name_] = stream;
    }
    return stream_map_.at(current_stream_name_);
  }

 private:
  OpKernelContext *ctx_;
  std::string current_stream_name_;
  std::unordered_map<std::string, cudaStream_t> stream_map_;
};
}  // namespace tf_internal