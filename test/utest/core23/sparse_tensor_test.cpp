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

#include <sparse_tensor.hpp>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR::core23;
using namespace HugeCTR;
void test_impl() {
  Shape shape({128, 30});
  int64_t slot_num = 26;
  SparseTensor23 sp_tensor(shape, ScalarType::UInt64, slot_num);
  Tensor val_tensor = sp_tensor.get_value_tensor();
  Tensor offset_tensor = sp_tensor.get_rowoffset_tensor();
  auto num_off = offset_tensor.num_elements();
  void* val = sp_tensor.get_value_ptr();
  EXPECT_FALSE(val == nullptr);
  EXPECT_TRUE(sp_tensor.nnz() == 0);
  EXPECT_TRUE(sp_tensor.max_nnz() == shape.size());
  EXPECT_TRUE(shape == val_tensor.shape());
  EXPECT_TRUE(ScalarType::Int32 == offset_tensor.data_type().type());
  EXPECT_TRUE(offset_tensor.data_type().match<int32_t>());

  std::cout << "Shape " << val_tensor.shape() << std::endl;
  std::cout << "num_off " << num_off << std::endl;
}

}  // namespace

TEST(test_core23, sparse_tensor) { test_impl(); }
