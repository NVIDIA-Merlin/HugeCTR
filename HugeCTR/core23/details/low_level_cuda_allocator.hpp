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

#include <cuda.h>

#include <core23/allocator.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

namespace core23 {

class Device;

class LowLevelCUDAAllocator : public Allocator {
 public:
  LowLevelCUDAAllocator(const Device& device, bool compressible = false);
  ~LowLevelCUDAAllocator() override;

  void* allocate(int64_t size, CUDAStream) override;

  [[nodiscard]] void* resize(void* ptr, int64_t new_size, CUDAStream stream = CUDAStream()) {
    return can_resize() ? do_resize(ptr, new_size, stream) : nullptr;
  }
  bool can_resize() const { return true; };

  void deallocate(void* ptr, CUDAStream) override;

  int64_t default_alignment() const override;

 private:
  void* do_resize(void* ptr, int64_t new_size, CUDAStream);

  size_t get_padded_size(size_t size) const {
    return ((size + granularity_ - 1) / granularity_) * granularity_;
  }
  CUdeviceptr reserve(CUdeviceptr old_ptr, int64_t new_size);
  CUdeviceptr allocate_common(CUdeviceptr ptr, int64_t new_size);
  void deallocate_common(CUdeviceptr ptr);

  struct Range {
    CUdeviceptr start;
    size_t size;
  };
  struct AllocInfo {
    size_t reserve_size;
    size_t alloc_size;
    std::vector<CUmemGenericAllocationHandle> alloc_handles;
    std::vector<size_t> alloc_handle_sizes;
    std::vector<Range> va_ranges;
  };

  CUmemAllocationProp prop_;
  CUmemAccessDesc access_desc_;
  size_t granularity_;
  std::unordered_map<CUdeviceptr, AllocInfo> alloc_info_;
};

}  // namespace core23

}  // namespace HugeCTR
