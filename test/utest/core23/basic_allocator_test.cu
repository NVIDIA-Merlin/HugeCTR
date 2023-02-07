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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <base/debug/logger.hpp>
#include <common.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <cstdint>
#include <random>
#include <utils.cuh>

namespace {

using namespace HugeCTR::core23;

const AllocatorParams g_allocator_params = {};

template <typename T>
void launch_init_kernel_gpu(void* ptr, int64_t num_bytes) {
  int64_t length = num_bytes / sizeof(uint8_t);
  T val = 4;
  dim3 block(1024);
  dim3 grid(length / block.x);
  HugeCTR::initialize_array<<<grid, block>>>(reinterpret_cast<T*>(ptr), length, val);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
}

template <typename T>
void launch_init_kernel_cpu(void* ptr, int64_t num_bytes) {
  int64_t length = num_bytes / sizeof(uint8_t);
  T val = 4;
  for (int64_t i = 0; i < length; i++) {
    reinterpret_cast<T*>(ptr)[i] = val;
  }
}

template <typename T>
void launch_init_kernel(void* ptr, int64_t num_bytes, DeviceType device_type) {
  if (device_type == DeviceType::GPU) {
    launch_init_kernel_gpu<T>(ptr, num_bytes);
  } else if (device_type == DeviceType::CPU) {
    launch_init_kernel_cpu<T>(ptr, num_bytes);
  } else if (device_type == DeviceType::UNIFIED) {
    launch_init_kernel_gpu<T>(ptr, num_bytes);
    launch_init_kernel_cpu<T>(ptr, num_bytes);
  } else {
    FAIL() << "Not Implemented DeviceType.";
  }
}
constexpr size_t ITER = 32;

void test_impl(AllocatorParams allocator_params, const Device& device) {
  HCTR_LIB_THROW(cudaFree(0));

  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int64_t> uniform_dist(1, 8);

  auto allocator = GetAllocator(allocator_params, device);
  const int64_t num_bytes = 1024 * 1024;
  const int64_t s = uniform_dist(e);
  auto ptr = allocator->allocate(num_bytes * s);
  EXPECT_FALSE(ptr == nullptr);
  EXPECT_TRUE(reinterpret_cast<intptr_t>(ptr) % allocator->default_alignment() == 0);
  launch_init_kernel<int8_t>(ptr, num_bytes * s, device.type());
}

}  // namespace

TEST(test_core23, allocator_simple_cuda) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::GPU, 0);
  test_impl(my_allocator_params, device);
}

TEST(test_core23, allocator_compressible_cuda) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::GPU, 0);
  my_allocator_params.compressible = true;
  test_impl(my_allocator_params, device);
}

TEST(test_core23, allocator_pool_cuda) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::GPU, 0);
  // TODO: change this line after introducing the ResourceManager
  my_allocator_params.custom_factory = [](const auto& params, const auto& device) {
    return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
  };
  test_impl(my_allocator_params, device);
}

TEST(test_core23, allocator_pinned_host) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::CPU);
  test_impl(my_allocator_params, device);
}

TEST(test_core23, allocator_new_delete) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::CPU);
  my_allocator_params.pinned = false;
  test_impl(my_allocator_params, device);
}

TEST(test_core23, allocator_managed_cuda) {
  AllocatorParams my_allocator_params = g_allocator_params;
  Device device(DeviceType::UNIFIED, 0);
  test_impl(my_allocator_params, device);
}
