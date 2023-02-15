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
#include <core23/details/simple_cuda_allocator.hpp>
#include <core23/macros.hpp>

namespace HugeCTR {

namespace core23 {

void* SimpleCUDAAllocator::allocate(int64_t size, CUDAStream) {
  void* ptr;
  HCTR_LIB_THROW(cudaMalloc(&ptr, size));
  return ptr;
}

void SimpleCUDAAllocator::deallocate(void* ptr, CUDAStream) { HCTR_LIB_THROW(cudaFree(ptr)); }

int64_t SimpleCUDAAllocator::default_alignment() const { return kcudaAllocationAlignment; }

}  // namespace core23

}  // namespace HugeCTR
