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
#include <nccl.h>

#include <core/macro.hpp>
#include <memory>

namespace core {
class GPUResourceBase {
 public:
  virtual ~GPUResourceBase() = default;
  virtual void set_stream(const std::string &name) = 0;
  virtual std::string get_current_stream_name() = 0;
  virtual cudaStream_t get_stream() = 0;  // will return current stream
};

class CoreResourceManager {
 public:
  virtual ~CoreResourceManager() = default;

  virtual std::shared_ptr<GPUResourceBase> get_local_gpu() = 0;

  virtual const ncclComm_t &get_nccl() const = 0;

  virtual int get_local_gpu_id() const = 0;

  virtual int get_global_gpu_id() const = 0;

  virtual int get_device_id() const = 0;

  virtual size_t get_local_gpu_count() const = 0;

  virtual size_t get_global_gpu_count() const = 0;

  virtual int get_gpu_global_id_from_local_id(int local_id) const = 0;

  virtual int get_gpu_local_id_from_global_id(int global_id) const = 0;
};

}  // namespace core