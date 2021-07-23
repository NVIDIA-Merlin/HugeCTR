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

#include "HugeCTR/include/layers/reshape_layer.hpp"

#include <math.h>

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {
template <typename T>
void reshape_test(vector<size_t>& dims_in, vector<size_t>& dims_out) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  Tensor2<T> in_tensor;
  buff->reserve(dims_in, &in_tensor);

  Tensor2<T> out_tensor;
  buff->reserve(dims_out, &out_tensor);
  ReshapeLayer<T> reshape_layer(in_tensor, out_tensor, buff, test::get_default_gpu());

  buff->allocate();
  reshape_layer.initialize();

  ASSERT_TRUE(out_tensor.allocated());

  std::vector<size_t> out_dims = out_tensor.get_dimensions();
  size_t n_in_elems = in_tensor.get_num_elements();
  size_t n_out_elems = out_tensor.get_num_elements();

  if (out_dims.size() == 2) {
    ASSERT_TRUE(out_dims.size() == 2 && n_in_elems == n_out_elems &&
                dims_out[1] == out_dims[out_dims.size() - 1]);
  } else {
    ASSERT_TRUE(out_dims.size() == 3 && n_in_elems == n_out_elems &&
                dims_out[1] == out_dims[out_dims.size() - 2] &&
                dims_out[2] == out_dims[out_dims.size() - 1]);
  }
}

template <typename T>
void reshape_2d_test(size_t dim0, size_t dim1, size_t leading_dim) {
  vector<size_t> in_dims = {dim0, dim1};
  vector<size_t> dims_out = {dim0 * dim1 / leading_dim, leading_dim};
  reshape_test<T>(in_dims, dims_out);
}

template <typename T>
void reshape_3d_test(size_t dim0, size_t dim1, size_t dim2, size_t leading_dim) {
  vector<size_t> in_dims = {dim0, dim1, dim2};
  vector<size_t> dims_out = {dim0 * dim1 * dim2 / leading_dim, leading_dim};
  reshape_test<T>(in_dims, dims_out);
}

template <typename T>
void reshape_2d_to_3d_test(size_t dim0, size_t leading_dim, size_t time_step) {
  vector<size_t> dims_in = {dim0, leading_dim};
  if (dim0 % time_step != 0)
    throw std::runtime_error("Error: the input first dimension is not divisible by time step");
  else {
    vector<size_t> dims_out = {dim0 / time_step, time_step, leading_dim};
    reshape_test<T>(dims_in, dims_out);
  }
}

}  // namespace

TEST(reshape_layer, fp32_32x320to8x1280) { reshape_2d_test<float>(8 * 4, 320, 4 * 320); }

TEST(reshape_layer, fp32_320x32to32x320) { reshape_2d_test<float>(320, 32, 320); }

TEST(reshape_layer, fp32_320x32to16x1280) { reshape_2d_test<float>(320, 32, 1280); }

TEST(reshape_layer, fp32_320x7x13to7x13x320) { reshape_2d_test<float>(320, 7 * 13, 320); }

TEST(reshape_layer, fp32_32x320to16x640) { reshape_2d_test<float>(8 * 4, 320, 2 * 320); }

TEST(reshape_layer, fp32_8x4x320to8x1280) { reshape_3d_test<float>(8, 4, 320, 4 * 320); }

TEST(reshape_layer, fp32_8x4x320to4x2560) { reshape_3d_test<float>(8, 4, 320, 8 * 320); }

TEST(reshape_layer, fp32_8x4x320to32x2560) { reshape_3d_test<float>(8, 4, 320, 320); }
TEST(reshape_layer, fp32_8x100x4to8x100x4) { reshape_2d_to_3d_test<float>(8 * 100 * 4, 4, 100); }

TEST(reshape_layer, fp16_32x320to8x1280) { reshape_2d_test<__half>(8 * 4, 320, 4 * 320); }

TEST(reshape_layer, fp16_32x320to16x640) { reshape_2d_test<__half>(8 * 4, 320, 2 * 320); }

TEST(reshape_layer, fp16_8x4x320to8x1280) { reshape_3d_test<__half>(8, 4, 320, 4 * 320); }

TEST(reshape_layer, fp16_8x4x320to4x2560) { reshape_3d_test<__half>(8, 4, 320, 8 * 320); }

TEST(reshape_layer, fp16_8x4x320to32x2560) { reshape_3d_test<__half>(8, 4, 320, 320); }