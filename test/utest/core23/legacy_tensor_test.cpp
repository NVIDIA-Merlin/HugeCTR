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
#include <core23/buffer_channel.hpp>
#include <core23/buffer_params.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_params.hpp>
#include <cstdint>

namespace {

using namespace HugeCTR::core23;

const BufferParams g_buffer_params = {};
const AllocatorParams g_allocator_params = {};

void test_impl(BufferParams buffer_params, AllocatorParams allocator_params) {
  Device device(DeviceType::GPU, 0);
  TensorParams tensor_params = TensorParams()
                                   .data_type(ScalarType::Int32)
                                   .allocator_params(allocator_params)
                                   .buffer_params(buffer_params)
                                   .device(device)
                                   .shape({1024, 128});

  // 1. Create the first, unique Tensor
  Tensor tensor0(tensor_params);

  EXPECT_TRUE(tensor_params.data_type() == tensor0.data_type());
  EXPECT_TRUE(tensor_params.device() == tensor0.device());
  EXPECT_TRUE(tensor_params.shape() == tensor0.shape());

  EXPECT_FALSE(tensor0.empty());
  EXPECT_TRUE(tensor0.is_unique());
  EXPECT_TRUE(tensor0.own_data());

  // 2. Create a Tensor from the first Tensor
  Tensor tensor1 = tensor0;
  EXPECT_TRUE(tensor0.data_type() == tensor1.data_type());
  EXPECT_TRUE(tensor0.device() == tensor1.device());
  EXPECT_TRUE(tensor0.shape() == tensor1.shape());

  EXPECT_FALSE(tensor1.empty());
  EXPECT_FALSE(tensor0.is_unique());
  EXPECT_FALSE(tensor1.is_unique());
  EXPECT_TRUE(tensor1.own_data());

  // 3. Create a Tensor from the preallocated data
  auto allocator = GetAllocator(allocator_params, device);
  int64_t size = tensor1.num_bytes();
  void* data = allocator->allocate(size);

  auto tensor2 = Tensor::bind(data, tensor1.shape(), tensor1.data_type(), tensor1.device());
  EXPECT_TRUE(tensor1.data_type() == tensor2.data_type());
  EXPECT_TRUE(tensor1.device() == tensor2.device());
  EXPECT_TRUE(tensor1.shape() == tensor2.shape());

  EXPECT_FALSE(tensor2.empty());
  EXPECT_FALSE(tensor2.is_unique());
  EXPECT_FALSE(tensor2.own_data());

  // 4. Create a Tensor from the preallocated data
  // auto tensor3 = reshape(tensor1, {256, 4, 128});
  auto tensor3 = tensor1.reshape({256, 4, 128});
  EXPECT_TRUE(tensor1.data_type() == tensor3.data_type());
  EXPECT_TRUE(tensor1.device() == tensor3.device());
  EXPECT_FALSE(tensor1.shape() == tensor3.shape());

  EXPECT_FALSE(tensor3.empty());
  EXPECT_FALSE(tensor3.is_unique());
  EXPECT_TRUE(tensor3.own_data());

  // // 5. Create another, unique Tensor
  Tensor tensor4({1024, 30, 128}, ScalarType::Half, tensor_params);

  EXPECT_TRUE(tensor0.data_type() != tensor4.data_type());
  EXPECT_TRUE(tensor0.device() == tensor4.device());
  EXPECT_TRUE(tensor0.shape() != tensor4.shape());

  EXPECT_FALSE(tensor4.empty());
  EXPECT_TRUE(tensor4.is_unique());
  EXPECT_TRUE(tensor4.is_unique());
  EXPECT_TRUE(tensor4.own_data());

  // 6. Compare their data pointers
  EXPECT_TRUE(tensor0.data() == tensor1.data());
  EXPECT_TRUE(tensor1.data() != tensor2.data());
  EXPECT_TRUE(tensor1.data() == tensor3.data());
  EXPECT_TRUE(tensor1.data() != tensor4.data());

  // 8. Create an emtpy Tensor and then overwrite it with the tensor0
  Tensor tensor5;
  EXPECT_TRUE(tensor5.empty());
  EXPECT_THROW(tensor5.shape(), HugeCTR::internal_runtime_error);
  EXPECT_THROW(tensor5.data_type(), HugeCTR::internal_runtime_error);
  EXPECT_FALSE(tensor5.own_data());
  EXPECT_TRUE(tensor5.data() == nullptr);
  tensor5 = tensor0;
  EXPECT_FALSE(tensor5.empty());
  EXPECT_NO_THROW(tensor5.shape());
  EXPECT_NO_THROW(tensor5.data_type());
  EXPECT_TRUE(tensor5.own_data());
  EXPECT_FALSE(tensor5.data() == nullptr);

  // 7. Create a Tensor with shape, data_type, and params specified in its constructor.
  // After calling data(), it override with an existing Tensor
  Tensor tensor6({512, 256}, ScalarType::Float,
                 tensor_params.buffer_channel(GetRandomBufferChannel()));
  EXPECT_TRUE(tensor6.shape() == Shape({512, 256}));
  EXPECT_FALSE(tensor6.shape() == tensor_params.shape());
  EXPECT_TRUE(tensor6.data_type() == ScalarType::Float);
  EXPECT_FALSE(tensor6.data_type() == tensor_params.data_type());
  auto tensor6_data = tensor6.data();
  tensor6 = tensor0;
  EXPECT_FALSE(tensor6.shape() == Shape({512, 256}));
  EXPECT_TRUE(tensor6.shape() == tensor_params.shape());
  EXPECT_FALSE(tensor6.data_type() == ScalarType::Float);
  EXPECT_TRUE(tensor6.data_type() == tensor_params.data_type());
  EXPECT_FALSE(tensor6.data() == tensor6_data);
  EXPECT_TRUE(tensor6.data() == tensor0.data());

  // 8. Create a Tensor and override it wih a new Tensor before it calls data().
  // Then call data() from another Tensor which belongs to the same channel.
  auto tensor_params0 = TensorParams({1024, 256}).data_type(ScalarType::Int64);
  auto tensor_params1 = TensorParams({1024, 256}).data_type(ScalarType::Int64);
  Tensor tensor7(tensor_params0);
  Tensor tensor8(tensor_params0);
  Tensor tensor9(tensor_params1);
  tensor7 = tensor9;
  EXPECT_TRUE(tensor8.data() != nullptr);


  allocator->deallocate(data);
}

}  // namespace

TEST(test_core23, legacy_tensor_test_with_unitary_buffer) {
  AllocatorParams allocator_params = g_allocator_params;
  BufferParams buffer_params = g_buffer_params;

  buffer_params.channel = "LEGACY_TENSOR_TEST_UNITARY";
  test_impl(buffer_params, allocator_params);
}

TEST(test_core23, legacy_tensor_test_with_confederal_buffer) {
  AllocatorParams allocator_params = g_allocator_params;
  BufferParams buffer_params = g_buffer_params;

  buffer_params.unitary = false;
  buffer_params.channel = "LEGACY_TENSOR_TEST_CONFEDERAL";
  test_impl(buffer_params, allocator_params);
}
