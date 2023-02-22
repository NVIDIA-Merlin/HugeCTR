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

#include <base/debug/logger.hpp>
#include <core23/details/pinned_host_allocator.hpp>
#include <cstddef>

namespace HugeCTR {

namespace core23 {

void* PinnedHostAllocator::allocate(int64_t size, CUDAStream) {
  void* ptr;
  HCTR_LIB_THROW(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
  return ptr;
}

void PinnedHostAllocator::deallocate(void* ptr, CUDAStream) { HCTR_LIB_THROW(cudaFreeHost(ptr)); }

int64_t PinnedHostAllocator::default_alignment() const { return alignof(std::max_align_t); }

}  // namespace core23

}  // namespace HugeCTR
