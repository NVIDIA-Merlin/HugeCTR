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
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/device.hpp>

namespace HugeCTR {

namespace core23 {

PoolCUDAAllocator::PoolCUDAAllocator(const Device& device) {
  HCTR_THROW_IF(device == DeviceType::CPU, Error_t::IllegalCall,
                "PoolCUDAAllocator cannot be used for CPU");
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = device.index()};
  cudaMemPoolProps mem_pool_props = {.allocType = cudaMemAllocationTypePinned,
                                     .handleTypes = cudaMemHandleTypeNone,
                                     .location = location};
  int driver_version = 0;
  HCTR_LIB_THROW(cudaDriverGetVersion(&driver_version));
  constexpr auto min_async_version = 11050;
  HCTR_THROW_IF(driver_version < min_async_version, Error_t::CudaDriverError,
                "the async allocator is not compatible with cuda driver < 11.5");

  // TODO: this pool must be shared among allocators
  HCTR_LIB_THROW(cudaMemPoolCreate(&mem_pool_handle_, &mem_pool_props));

  size_t free = 0, total = 0;
  HCTR_LIB_THROW(cudaMemGetInfo(&free, &total));

  uint64_t threshold = total;
  HCTR_LIB_THROW(
      cudaMemPoolSetAttribute(mem_pool_handle_, cudaMemPoolAttrReleaseThreshold, &threshold));

  std::vector<cudaMemAccessDesc> descs;
  for (int64_t id = 0; id < Device::count(); id++) {
    int can_access_peer = 0;
    HCTR_LIB_THROW(cudaDeviceCanAccessPeer(&can_access_peer, device.index(), id));
    if (can_access_peer && device.index() != id) {
      cudaMemAccessDesc desc = {};
      desc.location.type = cudaMemLocationTypeDevice;
      desc.location.id = id;
      desc.flags = cudaMemAccessFlagsProtReadWrite;
      descs.push_back(desc);
    }
  }
  if (!descs.empty()) {
    HCTR_LIB_THROW(cudaMemPoolSetAccess(mem_pool_handle_, descs.data(), descs.size()));
  }

  const int64_t initial_pool_size = free / 2;
  auto* ptr = allocate(initial_pool_size, 0);
  deallocate(ptr, 0);
}

PoolCUDAAllocator::~PoolCUDAAllocator() { HCTR_LIB_THROW(cudaMemPoolDestroy(mem_pool_handle_)); }

void* PoolCUDAAllocator::allocate(int64_t size, CUDAStream stream) {
  void* ptr = nullptr;
  HCTR_LIB_THROW(cudaMallocFromPoolAsync(&ptr, size, mem_pool_handle_, stream()));
  return ptr;
}

void PoolCUDAAllocator::deallocate(void* ptr, CUDAStream stream) {
  HCTR_LIB_THROW(cudaFreeAsync(ptr, stream()));
}

int64_t PoolCUDAAllocator::default_alignment() const { return 256; }

}  // namespace core23

}  // namespace HugeCTR
