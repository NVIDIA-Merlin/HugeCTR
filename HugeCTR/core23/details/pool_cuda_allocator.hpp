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

#include <cuda_runtime_api.h>

#include <core23/allocator.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

namespace core23 {

class Device;

class PoolCUDAAllocator : public Allocator {
 public:
  PoolCUDAAllocator(const Device& device);
  ~PoolCUDAAllocator() override;

  void* allocate(int64_t size, CUDAStream stream) override;

  void deallocate(void* ptr, CUDAStream) override;

  int64_t default_alignment() const override;

 private:
  cudaMemPool_t mem_pool_handle_;
};

}  // namespace core23

}  // namespace HugeCTR
