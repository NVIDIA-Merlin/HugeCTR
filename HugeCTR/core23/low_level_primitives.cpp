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

#include <algorithm>
#include <base/debug/logger.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/details/host_launch_helpers.hpp>
#include <core23/device.hpp>
#include <core23/device_guard.hpp>
#include <core23/low_level_primitives.hpp>
#include <optional>

namespace HugeCTR {

namespace core23 {

void copy_cpu_to_cpu(void* dst_data, const void* src_data, int64_t num_bytes,
                     std::optional<CUDAStream> stream_or) {
  auto dst = static_cast<uint8_t*>(dst_data);
  auto src = static_cast<const uint8_t*>(src_data);
  if (stream_or) {
    CopyParams<uint8_t, uint8_t>* params = new CopyParams<uint8_t, uint8_t>(dst, src, num_bytes);
    HCTR_LIB_THROW(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream_or.value()),
                                      static_cast<cudaHostFn_t>(copy_wrapper), params));
  } else {
    std::copy(src, src + num_bytes, dst);
  }
}
void copy_gpu_common(void* dst_data, const void* src_data, int64_t num_bytes, cudaMemcpyKind kind,
                     std::optional<CUDAStream> stream_or) {
  if (stream_or) {
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_data, src_data, num_bytes, kind,
                                   static_cast<cudaStream_t>(stream_or.value())));
  } else {
    HCTR_LIB_THROW(cudaMemcpy(dst_data, src_data, num_bytes, kind));
  }
}

void copy_gpu_to_gpu(void* dst_data, const void* src_data, int64_t num_bytes,
                     std::optional<CUDAStream> stream_or) {
  copy_gpu_common(dst_data, src_data, num_bytes, cudaMemcpyDeviceToDevice, stream_or);
}

void copy_cpu_to_gpu(void* dst_data, const void* src_data, int64_t num_bytes,
                     std::optional<CUDAStream> stream_or) {
  copy_gpu_common(dst_data, src_data, num_bytes, cudaMemcpyHostToDevice, stream_or);
}

void copy_gpu_to_cpu(void* dst_data, const void* src_data, int64_t num_bytes,
                     std::optional<CUDAStream> stream_or) {
  copy_gpu_common(dst_data, src_data, num_bytes, cudaMemcpyDeviceToHost, stream_or);
}

inline void copy_common(void* dst_data, const void* src_data, int64_t num_bytes,
                        const Device& dst_device, const Device& src_device,
                        std::optional<CUDAStream> stream_or) {
  HCTR_THROW_IF(dst_data == nullptr || src_data == nullptr, HugeCTR::Error_t::IllegalCall,
                "src_data or dst_data is nullptr");

  if (dst_data == src_data) {
    return;
  }

  DeviceGuard device_guard(src_device.type() == DeviceType::CPU? dst_device : src_device);
  if (src_device.type() == dst_device.type()) {
    if (src_device.type() == DeviceType::CPU) {
      copy_cpu_to_cpu(dst_data, src_data, num_bytes, stream_or);
    } else {
      copy_gpu_to_gpu(dst_data, src_data, num_bytes, stream_or);
    }
  } else {
    if (src_device.type() == DeviceType::CPU) {
      copy_cpu_to_gpu(dst_data, src_data, num_bytes, stream_or);
    } else {
      copy_gpu_to_cpu(dst_data, src_data, num_bytes, stream_or);
    }
  }
}
void copy_sync(void* dst_data, const void* src_data, int64_t num_bytes, const Device& dst_device,
               const Device& src_device) {
  copy_common(dst_data, src_data, num_bytes, dst_device, src_device, {});
}

void copy_async(void* dst_data, const void* src_data, int64_t num_bytes, const Device& dst_device,
                const Device& src_device, CUDAStream stream) {
  copy_common(dst_data, src_data, num_bytes, dst_device, src_device, stream);
}

}  // namespace core23
}  // namespace HugeCTR
