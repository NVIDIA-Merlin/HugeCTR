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

#include <array>
#include <base/debug/logger.hpp>
#include <common.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/buffer_client.hpp>
#include <core23/buffer_factory.hpp>
#include <core23/buffer_params.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/offsetted_buffer.hpp>
#include <cstdint>
#include <memory>
#include <random>

namespace {

using namespace HugeCTR::core23;

constexpr int64_t MAX_BUFFER_CHANNEL = 16;

const BufferParams g_buffer_params = {};
const AllocatorParams g_allocator_params = {};

class DummyBufferClient : public BufferClient {
 public:
  void* data() { return offsetted_buffer()->data(); }
};

void single_buffer_test_impl(BufferParams buffer_params, AllocatorParams allocator_params,
                             Device device) {
  std::random_device r;

  // To randomly choose a base data type size and alignment from the specified lists
  constexpr size_t n_type_sizes = 5;
  std::array<int64_t, n_type_sizes> type_size_list = {sizeof(char), sizeof(__half), sizeof(float),
                                                      sizeof(float2), sizeof(float4)};
  std::array<int64_t, n_type_sizes> alignment_list = {1, 2, 4, 8, 16};
  std::default_random_engine e0(r());
  std::uniform_int_distribution<int64_t> uniform_dist0(0, n_type_sizes - 1);

  // To randomly generate num_bytes in a specified range
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int64_t> uniform_dist1(1, 1024);

  // A Buffer is created
  auto buffer = GetBuffer(buffer_params, device, std::move(GetAllocator(allocator_params, device)));

  // Pass 1. Check if the Buffer is subscribable
  EXPECT_TRUE(buffer->subscribable());

  // Subscribe the Buffer
  std::vector<std::shared_ptr<DummyBufferClient>> buffer_clients(8);
  std::vector<BufferRequirements> buffer_requirements;
  for (auto& c : buffer_clients) {
    c.reset(new DummyBufferClient());
    auto index = uniform_dist0(e0);
    BufferRequirements requirements = {.num_bytes = type_size_list[index] * uniform_dist1(e1),
                                       .alignment = alignment_list[index]};
    EXPECT_NO_THROW(buffer->subscribe(c.get(), requirements));
    buffer_requirements.push_back(requirements);
  }

  // Pass 2. Check if the addresses are consistent with the requirements
  auto it_req = buffer_requirements.begin();
  for (auto& c : buffer_clients) {
    EXPECT_NO_THROW(c->data());
    EXPECT_TRUE(reinterpret_cast<std::intptr_t>(c->data()) % it_req->alignment == 0);
    it_req++;
  }

  // Pass 3. Check the behaviour of post-allocation subscription.
  std::shared_ptr<BufferClient> another_buffer_client(new DummyBufferClient());
  BufferRequirements another_requirement = {.num_bytes = type_size_list[0] * 16,
                                            .alignment = alignment_list[0]};
  if (buffer->subscribable()) {
    EXPECT_NO_THROW(buffer->subscribe(another_buffer_client.get(), another_requirement));
  } else {
    EXPECT_THROW(buffer->subscribe(another_buffer_client.get(), another_requirement),
                 HugeCTR::internal_runtime_error);
  }

  // Pass 4. Check the behaviour of post-allocation allocation.
  if (buffer->allocatable()) {
    std::vector<void*> buffer_client_data;
    for (auto buffer_client : buffer_clients) {
      buffer_client_data.push_back(buffer_client->data());
    }
    EXPECT_NO_THROW(buffer->allocate());
    for (int64_t i = 0; i < buffer_clients.size(); i++) {
      EXPECT_TRUE(buffer_client_data[i] == buffer_clients[i]->data());
    }
  } else {
    EXPECT_THROW(buffer->allocate(), HugeCTR::internal_runtime_error);
  }
}

}  // namespace

TEST(test_core23, single_unitary_buffer_gpu_simple) {
  Device device(DeviceType::GPU, 0);
  AllocatorParams my_allocator_params = g_allocator_params;
  single_buffer_test_impl(g_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_unitary_buffer_gpu_compressible) {
  Device device(DeviceType::GPU, 0);
  AllocatorParams my_allocator_params = g_allocator_params;
  my_allocator_params.compressible = true;
  single_buffer_test_impl(g_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_unitary_buffer_pinned_host) {
  Device device(DeviceType::CPU);
  AllocatorParams my_allocator_params = g_allocator_params;
  single_buffer_test_impl(g_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_unitary_buffer_new_delete) {
  Device device(DeviceType::CPU);
  AllocatorParams my_allocator_params = g_allocator_params;
  my_allocator_params.pinned = false;
  single_buffer_test_impl(g_buffer_params, my_allocator_params, device);
}

TEST(test_core23, multiple_unitary_buffer_gpu_simpple) {
  Device device(DeviceType::GPU, 0);
  AllocatorParams my_allocator_params = g_allocator_params;
  for (int64_t c = 0; c < MAX_BUFFER_CHANNEL; c++) {
    BufferParams my_buffer_params = g_buffer_params;
    my_buffer_params.channel = GetRandomBufferChannel();
    single_buffer_test_impl(my_buffer_params, my_allocator_params, device);
  }
}

TEST(test_core23, single_confederal_buffer_gpu_simple) {
  Device device(DeviceType::GPU, 0);
  AllocatorParams my_allocator_params = g_allocator_params;
  BufferParams my_buffer_params = g_buffer_params;
  my_buffer_params.unitary = false;
  single_buffer_test_impl(my_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_confederal_buffer_pinned_host) {
  Device device(DeviceType::CPU);
  AllocatorParams my_allocator_params = g_allocator_params;
  BufferParams my_buffer_params = g_buffer_params;
  my_buffer_params.unitary = false;
  single_buffer_test_impl(my_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_confederal_buffer_new_delete) {
  Device device(DeviceType::CPU);
  AllocatorParams my_allocator_params = g_allocator_params;
  my_allocator_params.pinned = false;
  BufferParams my_buffer_params = g_buffer_params;
  my_buffer_params.unitary = false;
  single_buffer_test_impl(my_buffer_params, my_allocator_params, device);
}

TEST(test_core23, single_confederal_buffer_gpu_pool) {
  Device device(DeviceType::GPU, 0);
  AllocatorParams my_allocator_params = g_allocator_params;
  // TODO: change this line after introducing the ResourceManager
  my_allocator_params.custom_factory = [](const auto& params, const auto& device) {
    return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
  };
  BufferParams my_buffer_params = g_buffer_params;
  my_buffer_params.unitary = false;
  single_buffer_test_impl(my_buffer_params, my_allocator_params, device);
}
