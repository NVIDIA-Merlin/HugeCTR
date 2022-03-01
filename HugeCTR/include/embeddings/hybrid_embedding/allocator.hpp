/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "HugeCTR/include/common.hpp"

class CudaPreAllocator {
  void *ptr_;
  size_t size_;

 public:
  CudaPreAllocator() : ptr_(nullptr), size_(0) {}

  template <typename T>
  void reserve(const std::vector<size_t> &dimensions) {
    size_t s = sizeof(T);
    for (size_t dimension : dimensions) {
      s *= dimension;
    }
    size_ += s;
  }

  void pre_allocate() { HCTR_LIB_THROW(cudaMalloc(&ptr_, size_)); }

  void *allocate(size_t size) const {
    if (size > size_) {
      HCTR_OWN_THROW(Error_t::OutOfMemory, "Out of memory");
    }
    return ptr_;
  }
  void deallocate(void *ptr) const { HCTR_LIB_THROW(cudaFree(ptr)); }
};
