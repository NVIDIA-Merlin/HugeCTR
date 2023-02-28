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
#include <core23/details/new_delete_allocator.hpp>
#include <cstddef>

namespace HugeCTR {

namespace core23 {

void* NewDeleteAllocator::allocate(int64_t size, CUDAStream) { return ::operator new(size); }

void NewDeleteAllocator::deallocate(void* ptr, CUDAStream) { return ::operator delete(ptr); }

int64_t NewDeleteAllocator::default_alignment() const { return alignof(std::max_align_t); }

}  // namespace core23

}  // namespace HugeCTR
