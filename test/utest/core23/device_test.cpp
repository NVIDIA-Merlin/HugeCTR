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
#include <core23/device.hpp>
#include <functional>
#include <unordered_set>

namespace {

using namespace HugeCTR::core23;

void test_impl(DeviceType true_type, DeviceIndex index, DeviceType false_type) {
  Device device0(true_type, index);
  EXPECT_NO_THROW(HCTR_LOG_S(DEBUG, ROOT) << device0 << std::endl);

  EXPECT_TRUE(device0.type() == true_type);
  EXPECT_TRUE(device0.index() == index);

  Device device1(true_type, index);
  Device device2(true_type, index + 1);
  Device device3(false_type, index);

  EXPECT_TRUE(device0 == device1);
  if (true_type == DeviceType::CPU) {
    EXPECT_TRUE(device0 == device2);
  } else {
    EXPECT_FALSE(device0 == device2);
  }
  EXPECT_FALSE(device0 == device3);

  int raw_device_id = 0;
  cudaGetDevice(&raw_device_id);
  Device device4(DeviceType::GPU, raw_device_id);
  auto current_device = Device::current();
  EXPECT_TRUE(current_device == device4);
}

}  // namespace

TEST(test_core23, gpu_device_test) { test_impl(DeviceType::GPU, 0, DeviceType::CPU); }
TEST(test_core23, cpu_device_test) { test_impl(DeviceType::CPU, 0, DeviceType::GPU); }
TEST(test_core23, unified_device_test) { test_impl(DeviceType::UNIFIED, 0, DeviceType::GPU); }
