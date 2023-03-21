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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/cuda_stream.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/logger.hpp>
#include <core23/low_level_primitives.hpp>
#include <cstdint>
#include <functional>
#include <utest/test_utils.hpp>
#include <vector>

namespace {

using namespace HugeCTR::core23;

constexpr size_t NUM_STREAMS = 8;
constexpr size_t NUM_ITERS = 8;
constexpr int64_t NUM_ELEMENTS = 1024 * 1024;
constexpr int64_t NUM_BYTES = NUM_ELEMENTS * sizeof(size_t);

void multi_stream_allocator_test_impl() {
  AllocatorParams allocator_params;
  Device device(DeviceType::GPU, 0);

  auto legacy_allocator = GetAllocator(allocator_params, device);
  std::vector<std::unique_ptr<size_t, std::function<void(size_t*)>>> d_outs;
  std::vector<std::vector<size_t>> h_outs;
  std::vector<std::vector<size_t>> h_refs;

  for (size_t sid = 0; sid < NUM_STREAMS; sid++) {
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
      d_outs.emplace_back(
          []() {
            size_t* ptr = nullptr;
            HCTR_LIB_THROW(cudaMalloc(&ptr, NUM_BYTES));
            return ptr;
          }(),
          [](size_t* ptr) { HCTR_LIB_THROW(cudaFree(ptr)); });
      h_outs.emplace_back(NUM_ELEMENTS);
      h_refs.emplace_back(NUM_ELEMENTS, sid * NUM_ITERS + iter);
    }
  }

  // TODO: change this line after introducing the ResourceManager
  allocator_params.custom_factory = [](const AllocatorParams& params,
                                       const Device& device) -> std::unique_ptr<Allocator> {
    return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
  };
  auto pool_allocator = GetAllocator(allocator_params, device);
  std::vector<CUDAStream> stream_vector;
  for (size_t sid = 0; sid < NUM_STREAMS; sid++) {
    auto stream = CUDAStream(cudaStreamDefault);
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
      size_t* ptr = static_cast<size_t*>(pool_allocator->allocate(NUM_BYTES, stream));
      EXPECT_FALSE(ptr == nullptr);

      fill_async(ptr, NUM_ELEMENTS, sid * NUM_ITERS + iter, device, stream);

      HCTR_LIB_THROW(cudaMemcpyAsync(d_outs[sid * NUM_ITERS + iter].get(), ptr, NUM_BYTES,
                                     cudaMemcpyDeviceToDevice, stream()));
      EXPECT_NO_THROW(pool_allocator->deallocate(ptr, stream));
    }
    stream_vector.push_back(stream);
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (size_t sid = 0; sid < NUM_STREAMS; sid++) {
    auto stream = stream_vector[sid];
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
      HCTR_LIB_THROW(cudaMemcpyAsync(h_outs[sid * NUM_ITERS + iter].data(),
                                     d_outs[sid * NUM_ITERS + iter].get(), NUM_BYTES,
                                     cudaMemcpyDeviceToHost, stream()));
    }
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (size_t sid = 0; sid < NUM_STREAMS; sid++) {
    for (size_t iter = 0; iter < NUM_ITERS; iter++) {
      ASSERT_TRUE(HugeCTR::test::compare_array_approx<size_t>(h_outs[sid * NUM_ITERS + iter].data(),
                                                              h_refs[sid * NUM_ITERS + iter].data(),
                                                              NUM_ELEMENTS, 0));
    }
  }
}

}  // namespace

TEST(test_core23, multi_stream_allocator_cuda_pool) { multi_stream_allocator_test_impl(); }
