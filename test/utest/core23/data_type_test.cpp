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
#include <core23/data_type.hpp>
#include <functional>
#include <unordered_set>

namespace {

using namespace HugeCTR::core23;

template <ScalarType TrueType, ScalarType FalseType>
void test_impl() {
  static_assert(TrueType != FalseType);

  DataType data_type0(TrueType);
  EXPECT_NO_THROW(HCTR_LOG_S(DEBUG, ROOT) << data_type0 << std::endl);

  EXPECT_TRUE(data_type0.match<typename ToBuiltInType<TrueType>::value>());
  EXPECT_FALSE(data_type0.match<typename ToBuiltInType<FalseType>::value>());

  EXPECT_TRUE(data_type0.size() == sizeof(typename ToBuiltInType<TrueType>::value));

  DataType data_type1(TrueType);
  EXPECT_TRUE(data_type0 == data_type1);
  EXPECT_FALSE(data_type0 != data_type1);

  DataType data_type2(FalseType);
  EXPECT_TRUE(data_type0 != data_type2);
  EXPECT_FALSE(data_type0 == data_type2);

  std::unordered_set<DataType> types = {data_type0, data_type1, data_type2};
  EXPECT_TRUE(types.size() == 2);
}

}  // namespace

TEST(test_core23, data_type_float_test) { test_impl<ScalarType::Float, ScalarType::Double>(); }
TEST(test_core23, data_type_half_test) { test_impl<ScalarType::Half, ScalarType::Double>(); }
TEST(test_core23, data_type_int8_test) { test_impl<ScalarType::Int8, ScalarType::Int64>(); }
TEST(test_core23, data_type_int32_test) { test_impl<ScalarType::Int32, ScalarType::Int64>(); }
