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

#include <base/debug/logger.hpp>
#include <common.hpp>
#include <core23/tensor_params.hpp>
#include <cstdint>
#include <random>

namespace {

using namespace HugeCTR::core23;

void tensor_params_test_impl() {
  TensorParams tensor_params0;
  EXPECT_THROW(tensor_params0.shape(), HugeCTR::internal_runtime_error);
  EXPECT_TRUE(tensor_params0.data_type() == DataType());
  EXPECT_TRUE(tensor_params0.device() == Device());
  EXPECT_TRUE(tensor_params0.stream() == CUDAStream());
  EXPECT_TRUE(tensor_params0.alignment() == 256);

  TensorParams tensor_params1 = tensor_params0.shape({32, 64});
  EXPECT_NO_THROW(tensor_params1.shape());
  EXPECT_TRUE(tensor_params1.shape() == Shape({32, 64}));
  EXPECT_TRUE(tensor_params0.data_type() == tensor_params1.data_type());
  EXPECT_TRUE(tensor_params0.device() == tensor_params1.device());
  EXPECT_TRUE(tensor_params0.stream() == tensor_params1.stream());
  EXPECT_TRUE(tensor_params0.alignment() == tensor_params1.alignment());

  TensorParams tensor_params2({32, 64});
  EXPECT_NO_THROW(tensor_params2.shape());
  EXPECT_TRUE(tensor_params1.shape() == tensor_params2.shape());
  EXPECT_TRUE(tensor_params1.data_type() == tensor_params2.data_type());
  EXPECT_TRUE(tensor_params1.device() == tensor_params2.device());
  EXPECT_TRUE(tensor_params1.stream() == tensor_params2.stream());
  EXPECT_TRUE(tensor_params1.alignment() == tensor_params2.alignment());

  CUDAStream stream0(cudaStreamDefault, 0);
  TensorParams tensor_params3 = tensor_params2.data_type(ScalarType::Half)
                                    .alignment(64)
                                    .device(DeviceType::CPU)
                                    .stream(stream0)
                                    .shape({16, 64});
  EXPECT_NO_THROW(tensor_params3.shape());
  EXPECT_FALSE(tensor_params2.shape() == tensor_params3.shape());
  EXPECT_FALSE(tensor_params2.data_type() == tensor_params3.data_type());
  EXPECT_FALSE(tensor_params2.device() == tensor_params3.device());
  EXPECT_FALSE(tensor_params2.stream() == tensor_params3.stream());
  EXPECT_FALSE(tensor_params2.alignment() == tensor_params3.alignment());

  EXPECT_TRUE(tensor_params3.shape() == Shape({16, 64}));
  EXPECT_TRUE(tensor_params3.data_type() == DataType(ScalarType::Half));
  EXPECT_TRUE(tensor_params3.device() == Device(DeviceType::CPU));
  EXPECT_TRUE(tensor_params3.stream() == stream0);
  EXPECT_TRUE(tensor_params3.alignment() == 64);

  BufferParams buffer_params;
  AllocatorParams allocator_params;
  TensorParams tensor_params4 =
      TensorParams().buffer_params(buffer_params).allocator_params(allocator_params);
  EXPECT_TRUE(buffer_params.channel == tensor_params4.buffer_params().channel);
  EXPECT_TRUE(buffer_params.unitary == tensor_params4.buffer_params().unitary);
  EXPECT_TRUE(allocator_params.compressible == tensor_params4.allocator_params().compressible);
  EXPECT_TRUE(allocator_params.pinned == tensor_params4.allocator_params().pinned);
}

}  // namespace

TEST(test_core23, tensor_params_test) { tensor_params_test_impl(); }
