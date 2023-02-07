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
#include <core23/curand_generator.hpp>
#include <core23/device.hpp>

namespace {

using namespace HugeCTR::core23;

void test_impl(Device device) {
  CURANDGenerator generator0(device);
  CURANDGenerator generator1(device);
  CURANDGenerator generator2 = generator0;
  EXPECT_FALSE(generator0 == generator1);
  EXPECT_TRUE(generator0 == generator2);
  EXPECT_FALSE(generator0() == generator1());
  EXPECT_TRUE(generator0() == generator2());
  EXPECT_FALSE(static_cast<curandGenerator_t>(generator0) ==
               static_cast<curandGenerator_t>(generator1));
  EXPECT_TRUE(static_cast<curandGenerator_t>(generator0) ==
              static_cast<curandGenerator_t>(generator2));

  EXPECT_NO_THROW(generator0.set_stream(CUDAStream()));
}

}  // namespace

TEST(test_core23, curand_generator_gpu_test) { test_impl(DeviceType::GPU); }
TEST(test_core23, curand_generator_cpu_test) { test_impl(DeviceType::CPU); }
