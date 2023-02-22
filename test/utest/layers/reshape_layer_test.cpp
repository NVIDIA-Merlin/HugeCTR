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

#include <gtest/gtest.h>

#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/reshape_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

template <typename T>
void reshape_test(core23::Shape& shape_in, core23::Shape& shape_out) {
  core23::Tensor bottom_tensor(core23::TensorParams().shape(shape_in));
  core23::Tensor top_tensor(core23::TensorParams().shape(shape_out));
  ReshapeLayer<T> reshape_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  reshape_layer.initialize();

  ASSERT_FALSE(top_tensor.empty());

  core23::Shape out_shape = top_tensor.shape();
  int64_t n_in_elems = bottom_tensor.num_elements();
  int64_t n_out_elems = top_tensor.num_elements();

  if (out_shape.dims() == 2) {
    ASSERT_TRUE(out_shape.dims() == 2 && n_in_elems == n_out_elems &&
                shape_out.size(1) == out_shape.size(out_shape.dims() - 1));
  } else {
    ASSERT_TRUE(out_shape.dims() == 3 && n_in_elems == n_out_elems &&
                shape_out.size(1) == out_shape.size(out_shape.dims() - 2) &&
                shape_out.size(2) == out_shape.size(out_shape.dims() - 1));
  }
}

template <typename T>
void reshape_2d_test(int64_t dim0, int64_t dim1, int64_t leading_dim) {
  core23::Shape in_dims = {dim0, dim1};
  core23::Shape shape_out = {dim0 * dim1 / leading_dim, leading_dim};
  reshape_test<T>(in_dims, shape_out);
}

template <typename T>
void reshape_3d_test(int64_t dim0, int64_t dim1, int64_t dim2, int64_t leading_dim) {
  core23::Shape in_dims = {dim0, dim1, dim2};
  core23::Shape shape_out = {dim0 * dim1 * dim2 / leading_dim, leading_dim};
  reshape_test<T>(in_dims, shape_out);
}

template <typename T>
void reshape_2d_to_3d_test(int64_t dim0, int64_t leading_dim, int64_t time_step) {
  core23::Shape shape_in = {dim0, leading_dim};
  if (dim0 % time_step != 0)
    throw std::runtime_error("Error: the input first dimension is not divisible by time step");
  else {
    core23::Shape shape_out = {dim0 / time_step, time_step, leading_dim};
    reshape_test<T>(shape_in, shape_out);
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