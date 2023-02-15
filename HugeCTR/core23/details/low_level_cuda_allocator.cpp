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
#include <core23/details/low_level_cuda_allocator.hpp>
#include <core23/device.hpp>
#include <core23/macros.hpp>

namespace HugeCTR {

namespace core23 {

LowLevelCUDAAllocator::LowLevelCUDAAllocator(const Device& device, bool compressible)
    : prop_{.type = CU_MEM_ALLOCATION_TYPE_PINNED,
            .location = {CU_MEM_LOCATION_TYPE_DEVICE, device.index()}},
      access_desc_{.location = prop_.location, .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
      granularity_(kcudaAllocationAlignment) {
  HCTR_THROW_IF(device.type() != DeviceType::GPU, HugeCTR::Error_t::IllegalCall,
                "Only DeviceType::GPU is supported.");

  HCTR_LIB_THROW(cudaFree(0));

  if (compressible) {
    int compression_supported = 0;
    HCTR_LIB_THROW(cuDeviceGetAttribute(
        &compression_supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device.index()));
    if (compression_supported) {
      prop_.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
    } else {
      HCTR_LOG_S(WARNING, ROOT) << "compressible memory is not supported on " << device
                                << std::endl;
    }
  }

  HCTR_LIB_THROW(
      cuMemGetAllocationGranularity(&granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
}

LowLevelCUDAAllocator::~LowLevelCUDAAllocator() {
  for (auto [ptr, alloc_info] : alloc_info_) {
    deallocate_common(ptr);
  }
}

CUdeviceptr LowLevelCUDAAllocator::reserve(CUdeviceptr old_ptr, int64_t new_size) {
  const auto reserve_size = (old_ptr == 0) ? 0ULL : alloc_info_[old_ptr].reserve_size;

  // Pretend it shrinks but does nothing
  if (new_size < reserve_size) {
    return old_ptr;
  }

  const size_t aligned_size = get_padded_size(new_size);
  CUdeviceptr new_ptr = 0ULL;
  // Try reserving a new VA range adajcent to the old VA range.
  HCTR_LIB_THROW(cuMemAddressReserve(&new_ptr, aligned_size - reserve_size, 0ULL,
                                     old_ptr + reserve_size, 0ULL));

  // the specific VA cannot be reserved due to some reasons
  if (old_ptr != 0ULL && new_ptr != (old_ptr + reserve_size)) {
    // Cancel the new VA rage reservation above
    HCTR_LIB_THROW(cuMemAddressFree(new_ptr, (aligned_size - reserve_size)));
    // Reserve a bigger VA range large enough to include both old and new allocations.
    HCTR_LIB_THROW(cuMemAddressReserve(&new_ptr, aligned_size, 0ULL, 0U, 0ULL));
    const auto alloc_size = alloc_info_[old_ptr].alloc_size;
    // Unmap the whole old physical memory allocation from the whole old VA range
    HCTR_LIB_THROW(cuMemUnmap(old_ptr, alloc_size));

    // Remap the old physical allocation to the new VA range
    CUdeviceptr ptr = new_ptr;
    auto alloc_handles = alloc_info_[old_ptr].alloc_handles;
    auto alloc_handle_sizes = alloc_info_[old_ptr].alloc_handle_sizes;
    for (size_t i = 0; i < alloc_handles.size(); i++) {
      const size_t handle_size = alloc_handle_sizes[i];
      HCTR_LIB_THROW(cuMemMap(ptr, handle_size, 0ULL, alloc_handles[i], 0ULL));
      HCTR_LIB_THROW(cuMemSetAccess(ptr, handle_size, &access_desc_, 1ULL));
      ptr += handle_size;
    }

    // Release the old VA range reservation
    for (size_t i = 0ULL; i < alloc_info_[old_ptr].va_ranges.size(); i++) {
      HCTR_LIB_THROW(cuMemAddressFree(alloc_info_[old_ptr].va_ranges[i].start,
                                      alloc_info_[old_ptr].va_ranges[i].size));
    }
    alloc_info_.erase(old_ptr);

    AllocInfo alloc_info = {aligned_size, 0, alloc_handles, alloc_handle_sizes, {}};
    old_ptr = new_ptr;
    alloc_info_[old_ptr] = alloc_info;
    alloc_info_[old_ptr].va_ranges.push_back({.start = new_ptr, .size = aligned_size});
  } else {
    if (old_ptr == 0ULL) {
      AllocInfo alloc_info = {aligned_size, 0, {}, {}, {}};
      old_ptr = new_ptr;
      alloc_info_[old_ptr] = alloc_info;
    }
    alloc_info_[old_ptr].va_ranges.push_back(
        {.start = new_ptr, .size = aligned_size - reserve_size});
  }
  return old_ptr;
}

void* LowLevelCUDAAllocator::allocate(int64_t size, CUDAStream) {
  return reinterpret_cast<void*>(allocate_common(0ULL, size));
}

CUdeviceptr LowLevelCUDAAllocator::allocate_common(CUdeviceptr old_ptr, int64_t new_size) {
  HCTR_THROW_IF(new_size < 0, HugeCTR::Error_t::IllegalCall, "The new_size is invalid");

  auto it = alloc_info_.find(old_ptr);
  HCTR_THROW_IF(old_ptr != 0ULL && it == alloc_info_.end(), HugeCTR::Error_t::IllegalCall,
                "`old_ptr` is not a nullptr but not under management of the allocator");

  const size_t alloc_size = (old_ptr == 0ULL) ? 0ULL : alloc_info_[old_ptr].alloc_size;

  if (new_size <= alloc_size) {
    return old_ptr;
  }

  const size_t size_diff = new_size - alloc_size;
  const size_t sz = get_padded_size(size_diff);

  auto new_ptr = reserve(old_ptr, alloc_size + sz);

  // Create a new physical memory allocation with the proper granularity
  CUmemGenericAllocationHandle alloc_handle;
  HCTR_LIB_THROW(cuMemCreate(&alloc_handle, sz, &prop_, 0));
  CUmemAllocationProp allocation_prop = {};
  cuMemGetAllocationPropertiesFromHandle(&allocation_prop, alloc_handle);

  if (allocation_prop.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC) {
    HCTR_LOG_S(DEBUG, ROOT) << "Obtained compressible memory allocation" << std::endl;
  }

  // Map the new physical memory allocation to an extended VA range
  HCTR_LIB_THROW(cuMemMap(new_ptr + alloc_size, sz, 0ULL, alloc_handle, 0ULL));

  // Allow read & write to the mapped region
  HCTR_LIB_THROW(cuMemSetAccess(new_ptr + alloc_size, sz, &access_desc_, 1));

  auto& alloc_info = alloc_info_[new_ptr];
  alloc_info.alloc_handles.push_back(alloc_handle);
  alloc_info.alloc_handle_sizes.push_back(sz);
  alloc_info.alloc_size = alloc_size + sz;

  return new_ptr;
}

void* LowLevelCUDAAllocator::do_resize(void* ptr, int64_t new_size, CUDAStream) {
  return reinterpret_cast<void*>(allocate_common(reinterpret_cast<CUdeviceptr>(ptr), new_size));
}

void LowLevelCUDAAllocator::deallocate(void* ptr, CUDAStream) {
  CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(ptr);
  deallocate_common(dev_ptr);
  alloc_info_.erase(dev_ptr);
}
void LowLevelCUDAAllocator::deallocate_common(CUdeviceptr dev_ptr) {
  auto it = alloc_info_.find(dev_ptr);
  HCTR_THROW_IF(dev_ptr == 0 || it == alloc_info_.end(), HugeCTR::Error_t::IllegalCall,
                "`ptr` is nullptr or not allocated by this allocator");

  auto alloc_info = it->second;

  HCTR_LIB_THROW(cuMemUnmap(dev_ptr, alloc_info.alloc_size));
  for (auto va_range : alloc_info.va_ranges) {
    HCTR_LIB_THROW(cuMemAddressFree(va_range.start, va_range.size));
  }
  for (auto alloc_handle : alloc_info.alloc_handles) {
    HCTR_LIB_THROW(cuMemRelease(alloc_handle));
  }
}

int64_t LowLevelCUDAAllocator::default_alignment() const { return granularity_; }

}  // namespace core23

}  // namespace HugeCTR
