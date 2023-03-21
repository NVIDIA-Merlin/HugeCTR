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

#include <core23/buffer_channel.hpp>
#include <core23/buffer_channel_helpers.hpp>
#include <core23/logger.hpp>

namespace {

using namespace HugeCTR::core23;

void test_impl() {
  BufferChannel buffer_channel0("funny_name");
  BufferChannel buffer_channel1("boring_name");
  EXPECT_FALSE(buffer_channel0 == buffer_channel1);
  EXPECT_TRUE(buffer_channel0 != buffer_channel1);

  BufferChannel buffer_channel2("funny_name");
  EXPECT_TRUE(buffer_channel0 == buffer_channel2);
  EXPECT_FALSE(buffer_channel0 != buffer_channel2);

  int64_t raw_channel = buffer_channel0;
  EXPECT_TRUE(raw_channel == buffer_channel0());
  EXPECT_FALSE(raw_channel == buffer_channel1());

  BufferChannel lhs = GetRandomBufferChannel();
  for (int64_t i = 0; i < 256; i++) {
    BufferChannel rhs = GetRandomBufferChannel();
    EXPECT_FALSE(lhs == rhs);
  }
}

}  // namespace

TEST(test_core23, buffer_channel_test) { test_impl(); }
