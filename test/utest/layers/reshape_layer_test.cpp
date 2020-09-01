/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <math.h>
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
void reshape_test(vector<size_t>& in_dims, size_t leading_dim) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  Tensor2<T> in_tensor;
  buff->reserve(in_dims, &in_tensor);

  Tensor2<T> out_tensor;
  ReshapeLayer<T> reshape_layer(in_tensor, in_tensor, out_tensor, buff, leading_dim,
                                test::get_default_gpu());

  buff->allocate();
  reshape_layer.initialize();

  ASSERT_TRUE(out_tensor.allocated());

  std::vector<size_t> out_dims = out_tensor.get_dimensions();
  size_t n_in_elems = in_tensor.get_num_elements();
  size_t n_out_elems = out_tensor.get_num_elements();

  ASSERT_TRUE(out_dims.size() == 2 && n_in_elems == n_out_elems &&
              leading_dim == out_dims[out_dims.size() - 1]);
}

template <typename T>
void reshape_2d_test(size_t dim0, size_t dim1, size_t leading_dim) {
  vector<size_t> in_dims = {dim0, dim1};
  reshape_test<T>(in_dims, leading_dim);
}

template <typename T>
void reshape_3d_test(size_t dim0, size_t dim1, size_t dim2, size_t leading_dim) {
  vector<size_t> in_dims = {dim0, dim1, dim2};
  reshape_test<T>(in_dims, leading_dim);
}

}  // namespace

TEST(reshape_layer, fp32_32x320to8x1280) { reshape_2d_test<float>(8 * 4, 320, 4 * 320); }

TEST(reshape_layer, fp32_32x320to16x640) { reshape_2d_test<float>(8 * 4, 320, 2 * 320); }

TEST(reshape_layer, fp32_8x4x320to8x1280) { reshape_3d_test<float>(8, 4, 320, 4 * 320); }

TEST(reshape_layer, fp32_8x4x320to4x2560) { reshape_3d_test<float>(8, 4, 320, 8 * 320); }

TEST(reshape_layer, fp32_8x4x320to32x2560) { reshape_3d_test<float>(8, 4, 320, 320); }

TEST(reshape_layer, fp16_32x320to8x1280) { reshape_2d_test<__half>(8 * 4, 320, 4 * 320); }

TEST(reshape_layer, fp16_32x320to16x640) { reshape_2d_test<__half>(8 * 4, 320, 2 * 320); }

TEST(reshape_layer, fp16_8x4x320to8x1280) { reshape_3d_test<__half>(8, 4, 320, 4 * 320); }

TEST(reshape_layer, fp16_8x4x320to4x2560) { reshape_3d_test<__half>(8, 4, 320, 8 * 320); }

TEST(reshape_layer, fp16_8x4x320to32x2560) { reshape_3d_test<__half>(8, 4, 320, 320); }
