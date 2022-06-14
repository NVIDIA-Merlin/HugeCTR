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
#include <stddef.h>

#include "../datatype.hpp"
#include "../device.hpp"
#include "HugeCTR/include/base/debug/logger.hpp"

namespace hctr_internal {
using core::DataType;
using core::DeviceType;

struct Allocator {
  virtual ~Allocator() = default;

  virtual void *allocate(size_t size) const = 0;

  virtual void release(void *ptr) const = 0;
};

Allocator *GetAllocator(DeviceType t);

class HostAllocator final : public Allocator {
 public:
  void *allocate(size_t size) const override {
    void *ptr;
    HCTR_LIB_THROW(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
  }
  void release(void *ptr) const override { HCTR_LIB_THROW(cudaFreeHost(ptr)); }
};

class CudaManagedAllocator final : public Allocator {
 public:
  void *allocate(size_t size) const override {
    void *ptr;
    HCTR_LIB_THROW(cudaMallocManaged(&ptr, size));
    return ptr;
  }
  void release(void *ptr) const override { HCTR_LIB_THROW(cudaFree(ptr)); }
};

class CudaAllocator final : public Allocator {
 public:
  void *allocate(size_t size) const override {
    void *ptr;
    HCTR_LIB_THROW(cudaMalloc(&ptr, size));
    return ptr;
  }
  void release(void *ptr) const override { HCTR_LIB_THROW(cudaFree(ptr)); }
};

}  // namespace hctr_internal