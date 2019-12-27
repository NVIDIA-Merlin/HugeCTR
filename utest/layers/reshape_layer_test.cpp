
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#include "HugeCTR/include/layers/reshape_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <math.h>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

void reshape_test(vector<int>& in_dims, int leading_dim) {
  std::shared_ptr<GeneralBuffer<float>> buff(new GeneralBuffer<float>());
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(in_dims, buff,
      (in_dims.size() == 3)? TensorFormat_t::HSW : TensorFormat_t::HW));
  buff->init(0);

  std::shared_ptr<Tensor<float>> out_tensor;
  ReshapeLayer reshape_layer(in_tensor, out_tensor, leading_dim, 0);

  ASSERT_TRUE(out_tensor);

  std::vector<int> out_dims = out_tensor->get_dims();
  int n_in_elems = in_tensor->get_num_elements();
  int n_out_elems = out_tensor->get_num_elements();

  ASSERT_TRUE(out_dims.size() == 2 &&
              n_in_elems == n_out_elems &&
              leading_dim == out_dims[out_dims.size() - 1]);
}

void reshape_2d_test(int dim0, int dim1, int leading_dim) {
  vector<int> in_dims = {dim0, dim1};
  reshape_test(in_dims, leading_dim);
}

void reshape_3d_test(int dim0, int dim1, int dim2, int leading_dim) {
  vector<int> in_dims = {dim0, dim1, dim2};
  reshape_test(in_dims, leading_dim);
}

}  // namespace

TEST(reshape_layer, 32x320to8x1280) {
  reshape_2d_test(8*4, 320, 4*320);
}

TEST(reshape_layer, 32x320to16x640) {
  reshape_2d_test(8*4, 320, 2*320);
}

TEST(reshape_layer, 8x4x320to8x1280) {
  reshape_3d_test(8, 4, 320, 4*320);
}

TEST(reshape_layer, 8x4x320to4x2560) {
  reshape_3d_test(8, 4, 320, 8*320);
}

TEST(reshape_layer, 8x4x320to32x2560) {
  reshape_3d_test(8, 4, 320, 320);
}
