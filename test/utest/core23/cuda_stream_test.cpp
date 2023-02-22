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
#include <core23/cuda_stream.hpp>

namespace {

using namespace HugeCTR::core23;

void test_impl() {
  CUDAStream default_stream;
  EXPECT_TRUE(default_stream() == 0);
  EXPECT_TRUE(static_cast<cudaStream_t>(default_stream) == default_stream());

  CUDAStream stream0 = CUDAStream(cudaStreamDefault, 0);
  CUDAStream stream1 = stream0;
  EXPECT_TRUE(stream0 == stream1);
  EXPECT_TRUE(stream0() == stream1());
  EXPECT_TRUE(static_cast<cudaStream_t>(stream0) == static_cast<cudaStream_t>(stream1));
  EXPECT_TRUE(static_cast<cudaStream_t>(stream0) == stream1());
  EXPECT_TRUE(stream0() == static_cast<cudaStream_t>(stream1));

  CUDAStream stream2 = CUDAStream(cudaStreamDefault, 0);
  EXPECT_FALSE(stream0 == stream2);
  EXPECT_FALSE(stream0() == stream2());
  EXPECT_FALSE(static_cast<cudaStream_t>(stream0) == static_cast<cudaStream_t>(stream2));
  EXPECT_FALSE(static_cast<cudaStream_t>(stream0) == stream2());

  stream2 = stream0;
  EXPECT_TRUE(stream0 == stream2);
  EXPECT_TRUE(stream0() == stream2());
  EXPECT_TRUE(static_cast<cudaStream_t>(stream0) == static_cast<cudaStream_t>(stream2));
  EXPECT_TRUE(static_cast<cudaStream_t>(stream0) == stream2());
}

}  // namespace

TEST(test_core23, cuda_stream_test) { test_impl(); }
