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

#include <common.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/allocator_params.hpp>
#include <core23/buffer_params.hpp>
#include <core23/details/pool_cuda_allocator.hpp>
#include <core23/logger.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <core23/tensor_params.hpp>
#include <cstdint>
#include <random>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR::core23;

constexpr int64_t height = 1024;
constexpr int64_t width = 256;
constexpr int64_t ITER = 4;

void test_impl() {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, height * width);

  std::vector<int> h_ins(height * width);
  for (size_t i = 0; i < h_ins.size(); i++) {
    h_ins[i] = uniform_dist(e);
  }

  TensorParams tensor_params0 = TensorParams()
                                    .data_type(ScalarType::Int32)
                                    .device(Device(DeviceType::GPU, 0))
                                    .shape({height, width});

  Tensor input_tensor = Tensor(tensor_params0);

  Tensor output_tensor = Tensor(tensor_params0);

  AllocatorParams allocator_params;
  // TODO: change this line after introducing the ResourceManager
  allocator_params.custom_factory = [](const auto& params, const auto& device) {
    return std::unique_ptr<Allocator>(new PoolCUDAAllocator(device));
  };

  BufferParams buffer_params;
  buffer_params.channel = "DYNAMIC_TENSOR_TEST";
  buffer_params.unitary = false;

  CUDAStream stream;

  TensorParams tensor_params1 =
      tensor_params0.allocator_params(allocator_params).buffer_params(buffer_params).stream(stream);

  copy_async(input_tensor.data(), h_ins.data(), input_tensor.num_bytes(), input_tensor.device(),
             DeviceType::CPU, stream);
  for (int64_t i = 0; i < ITER; i++) {
    Tensor temp_tensor0(tensor_params1);
    copy_async(temp_tensor0, (i == 0) ? input_tensor : output_tensor, stream);
    Tensor temp_tensor1(tensor_params1);
    copy_async(temp_tensor1, temp_tensor0, stream);
    copy_async(output_tensor, temp_tensor1, stream);
  }
  std::vector<int> h_outs(height * width);
  copy_sync(h_outs.data(), output_tensor.data(), output_tensor.num_bytes(), DeviceType::CPU,
            output_tensor.device());

  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_outs.data(), h_ins.data(), h_outs.size(), 0));
}

}  // namespace

TEST(test_core23, dynamic_tensor_test) { test_impl(); }
