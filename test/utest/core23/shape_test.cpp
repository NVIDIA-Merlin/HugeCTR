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
#include <core23/shape.hpp>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR::core23;

void test_impl() {
  Shape shape0;
  EXPECT_FALSE(shape0.valid());

  Shape shape1(32);
  EXPECT_FALSE(shape1.valid());

  Shape shape2{1024, 32, 128};
  EXPECT_TRUE(shape2.valid());
  EXPECT_TRUE(shape2.dims() == 3);
  EXPECT_TRUE(shape2.size(0) == 1024);
  EXPECT_TRUE(shape2.size(1) == 32);
  EXPECT_TRUE(shape2.size(2) == 128);
  EXPECT_TRUE(shape2.size() == 1024 * 32 * 128);

  Shape shape3({1024, 32, 128});
  EXPECT_TRUE(shape3.valid());
  EXPECT_TRUE(shape2 == shape3);
  EXPECT_FALSE(shape2 != shape3);
  EXPECT_TRUE(shape1 != shape3);
  EXPECT_FALSE(shape1 == shape3);

  Shape shape4({128, 32, 64});
  EXPECT_TRUE(shape4.valid());
  EXPECT_TRUE(shape2 != shape4);
  EXPECT_FALSE(shape2 == shape4);
}

}  // namespace

TEST(test_core23, shape_test) { test_impl(); }
