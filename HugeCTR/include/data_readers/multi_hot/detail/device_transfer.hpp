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

#include <utils.hpp>

namespace HugeCTR {

class DeviceTransfer {
 public:
  DeviceTransfer(size_t upload_gpu, uint8_t* upload_src, uint8_t* upload_dst, size_t size_bytes)
      : upload_gpu_(upload_gpu),
        upload_src_(const_cast<uint8_t*>(upload_src)),
        upload_dst_(const_cast<uint8_t*>(upload_dst)),
        size_bytes_(size_bytes) {}

  size_t get_device_id() const { return upload_gpu_; }

  void execute(const cudaStream_t& stream) {
    HCTR_LIB_THROW(
        cudaMemcpyAsync(upload_dst_, upload_src_, size_bytes_, cudaMemcpyHostToDevice, stream));
  }

 private:
  size_t upload_gpu_;
  uint8_t* upload_src_;
  uint8_t* upload_dst_;
  size_t size_bytes_;
};

}  // namespace HugeCTR