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

#include <base/debug/logger.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/details/low_level_cuda_allocator.hpp>
#include <core23/details/managed_cuda_allocator.hpp>
#include <core23/details/new_delete_allocator.hpp>
#include <core23/details/pinned_host_allocator.hpp>
#include <core23/details/simple_cuda_allocator.hpp>
#include <memory>

namespace HugeCTR {

namespace core23 {

std::unique_ptr<Allocator> GetDefaultGPUAllocator(const AllocatorParams& allocator_params,
                                                  const Device& device) {
  std::unique_ptr<Allocator> ret;
  if (allocator_params.pinned) {
    if (allocator_params.compressible) {
      ret.reset(new LowLevelCUDAAllocator(device, true));
    } else {
      ret.reset(new SimpleCUDAAllocator());
    }
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "Unimplemented Allocator");
  }
  return ret;
}

std::unique_ptr<Allocator> GetDefaultCPUAllocator(const AllocatorParams& allocator_params,
                                                  const Device& device) {
  std::unique_ptr<Allocator> ret;
  if (!allocator_params.compressible) {
    if (allocator_params.pinned) {
      ret.reset(new PinnedHostAllocator());
    } else {
      ret.reset(new NewDeleteAllocator());
    }
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "Unimplemented Allocator");
  }
  return ret;
}

std::unique_ptr<Allocator> GetDefaultUnifiedAllocator(const AllocatorParams& allocator_params,
                                                      const Device& device) {
  std::unique_ptr<Allocator> ret;
  if (!allocator_params.compressible && allocator_params.pinned) {
    ret.reset(new ManagedCUDAAllocator());
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "Unimplemented Allocator");
  }
  return ret;
}

std::unique_ptr<Allocator> GetAllocator(const AllocatorParams& allocator_params,
                                        const Device& device) {
  std::unique_ptr<Allocator> allocator = allocator_params.custom_factory(allocator_params, device);
  if (allocator == nullptr) {
    switch (device.type()) {
      case DeviceType::GPU:
        allocator = std::move(GetDefaultGPUAllocator(allocator_params, device));
        break;
      case DeviceType::CPU:
        allocator = std::move(GetDefaultCPUAllocator(allocator_params, device));
        break;
      case DeviceType::UNIFIED:
        allocator = std::move(GetDefaultUnifiedAllocator(allocator_params, device));
        break;
    }
  }
  return allocator;
}

}  // namespace core23

}  // namespace HugeCTR
