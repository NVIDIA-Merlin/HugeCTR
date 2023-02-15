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

#include <core/hctr_impl/allocator.hpp>

namespace hctr_internal {

static Allocator *allocator_table[static_cast<int>(DeviceType::MAX_DEVICE_NUM)];

Allocator *GetAllocator(DeviceType t) { return allocator_table[static_cast<int>(t)]; }

template <DeviceType t>
struct AllocatorRegister_ {
  AllocatorRegister_(Allocator *allocator) { allocator_table[static_cast<int>(t)] = allocator; }
};

#define REGISTER_ALLOCATOR(t, a, name)                       \
  namespace {                                                \
  static AllocatorRegister_<t> allocator_register_##name(a); \
  };

static HostAllocator host_allocator;
static CudaManagedAllocator cuda_managed_allocator;
static CudaAllocator cuda_allocator;

REGISTER_ALLOCATOR(DeviceType::CPU, &host_allocator, cpu);
REGISTER_ALLOCATOR(DeviceType::GPU, &cuda_allocator, gpu);
REGISTER_ALLOCATOR(DeviceType::CPUGPU, &cuda_managed_allocator, cpugpu);

}  // namespace hctr_internal