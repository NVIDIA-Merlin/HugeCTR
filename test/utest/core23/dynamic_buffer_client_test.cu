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
#include <core23/buffer_client.hpp>
#include <core23/buffer_factory.hpp>
#include <core23/buffer_params.hpp>
#include <core23/cuda_primitives.cuh>
#include <core23/cuda_stream.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/logger.hpp>
#include <core23/offsetted_buffer.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <utest/test_utils.hpp>
#include <vector>

namespace {

using namespace HugeCTR::core23;

constexpr size_t NUM_ELEMENTS = 1024;
constexpr size_t NUM_BYTES = NUM_ELEMENTS * sizeof(int);

class DynamicDummyBufferClient : public BufferClient {
 public:
  DynamicDummyBufferClient(const AllocatorParams& allocator_params, const Device& device,
                           BufferParams& buffer_params, CUDAStream stream)
      : buffer_(GetBuffer(buffer_params, device, GetAllocator(allocator_params, device))),
        stream_(stream) {
    BufferRequirements requirements = {.num_bytes = NUM_BYTES};
    buffer_->subscribe(this, requirements);
  }
  ~DynamicDummyBufferClient() {}
  void* data() const { return offsetted_buffer()->data(); }

 private:
  std::shared_ptr<Buffer> buffer_;
  CUDAStream stream_;
};

void test_impl() {
  AllocatorParams allocator_params;
  Device device(DeviceType::GPU, 0);
  // TODO: change this line after introducing the ResourceManager
  allocator_params.custom_factory = [](const auto& params, const auto& device) {
    return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
  };

  BufferParams buffer_params;
  buffer_params.channel = "DYNAMIC_BUFFER_CLIENT_TEST";
  buffer_params.unitary = false;

  dim3 block(1024);
  dim3 grid(NUM_ELEMENTS / block.x);
  CUDAStream stream(cudaStreamDefault);

  auto op = [] __device__(int x) { return 2 * x; };

  auto src_client =
      std::make_shared<DynamicDummyBufferClient>(allocator_params, device, buffer_params, stream);

  fill_kernel<<<grid, block, 0, stream()>>>((int*)src_client->data(), NUM_ELEMENTS, 1);

  auto dst_client =
      std::make_shared<DynamicDummyBufferClient>(allocator_params, device, buffer_params, stream);

  {
    auto temp_client0 =
        std::make_shared<DynamicDummyBufferClient>(allocator_params, device, buffer_params, stream);

    transform_kernel<int, int><<<grid, block, 0, stream()>>>(
        (int*)temp_client0->data(), (int*)src_client->data(), NUM_ELEMENTS, op);

    auto temp_client1 =
        std::make_shared<DynamicDummyBufferClient>(allocator_params, device, buffer_params, stream);
    transform_kernel<int, int><<<grid, block, 0, stream()>>>(
        (int*)temp_client1->data(), (int*)temp_client0->data(), NUM_ELEMENTS, op);

    transform_kernel<int, int><<<grid, block, 0, stream()>>>(
        (int*)dst_client->data(), (int*)temp_client1->data(), NUM_ELEMENTS, op);
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::vector<int> h_outs(NUM_ELEMENTS);
  HCTR_LIB_THROW(cudaMemcpy(h_outs.data(), dst_client->data(), NUM_BYTES, cudaMemcpyDeviceToHost));

  std::vector<int> h_refs(NUM_ELEMENTS, 8);
  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_outs.data(), h_refs.data(), NUM_ELEMENTS, 0));
}

}  // namespace

TEST(test_core23, dynamic_buffer_client_test) { test_impl(); }
