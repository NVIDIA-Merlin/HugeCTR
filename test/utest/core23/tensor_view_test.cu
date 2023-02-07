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
#include <core23/cuda_primitives.cuh>
#include <core23/macros.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_view.hpp>
#include <cstdint>
#include <random>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR::core23;

constexpr int64_t Dims = 3;
constexpr int64_t INPUT_SIZE_X = 128;
constexpr int64_t INPUT_SIZE_Y = 96;
constexpr int64_t INPUT_SIZE_Z = 48;
constexpr size_t INPUT_SIZE = INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_SIZE_Z;
constexpr size_t INPUT_BYTES = INPUT_SIZE * sizeof(int);

constexpr int64_t OFFSET_X = 4;
constexpr int64_t OFFSET_Y = 8;
constexpr int64_t OFFSET_Z = 2;

constexpr int64_t OUTPUT_SIZE_X = INPUT_SIZE_X - OFFSET_X;
constexpr int64_t OUTPUT_SIZE_Y = INPUT_SIZE_Y - OFFSET_Y;
constexpr int64_t OUTPUT_SIZE_Z = INPUT_SIZE_Z - OFFSET_Z;
constexpr size_t OUTPUT_SIZE = OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z;
constexpr size_t OUTPUT_BYTES = OUTPUT_SIZE * sizeof(int);

void tensor_view_test_impl() {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, 1024);

  std::vector<int> h_ins(INPUT_SIZE);
  for (size_t i = 0; i < h_ins.size(); i++) {
    h_ins[i] = uniform_dist(e);
  }

  std::vector<int> h_outs(OUTPUT_SIZE);
  const auto& h_refs = h_ins;

  // 3D TensorView Test
  {
    BufferParams buffer_params;
    buffer_params.channel = "3D_TENSOR_VIEW_TEST";

    auto input_tensor_params = TensorParams()
                                   .shape({INPUT_SIZE_Z, INPUT_SIZE_Y, INPUT_SIZE_X})
                                   .buffer_params(buffer_params)
                                   .data_type(ScalarType::Int32);
    auto input_tensor_3d = Tensor(input_tensor_params);

    auto output_tensor_3d =
        Tensor(input_tensor_params.shape({OUTPUT_SIZE_Z, OUTPUT_SIZE_Y, OUTPUT_SIZE_X}));

    HCTR_LIB_THROW(
        cudaMemcpy(input_tensor_3d.data(), h_ins.data(), INPUT_BYTES, cudaMemcpyHostToDevice));

    auto input_tensor_view = input_tensor_3d.view<int, Dims>({OFFSET_Z, OFFSET_Y, OFFSET_X});
    auto output_tensor_view = output_tensor_3d.view<int, Dims>();

    EXPECT_FALSE(input_tensor_view.size(0) * input_tensor_view.size(1) *
                     input_tensor_view.size(2) ==
                 input_tensor_3d.num_elements());
    EXPECT_TRUE(output_tensor_view.size(0) * output_tensor_view.size(1) *
                    output_tensor_view.size(2) ==
                output_tensor_3d.num_elements());

    EXPECT_FALSE(input_tensor_3d.data() == &input_tensor_view[0][0][0]);
    EXPECT_TRUE(input_tensor_3d.data<int>() + (OFFSET_Z * (INPUT_SIZE_Y * INPUT_SIZE_X) +
                                               (OFFSET_Y * INPUT_SIZE_X) + OFFSET_X) ==
                &input_tensor_view[0][0][0]);
    EXPECT_TRUE(output_tensor_3d.data() == &output_tensor_view[0][0][0]);

    auto output_shape = output_tensor_3d.shape();
    dim3 block(32, 32);
    dim3 grid((output_shape.size(2) + block.x - 1) / block.x,
              (output_shape.size(1) + block.y - 1) / block.y);
    copy_kernel<int><<<grid, block>>>(input_tensor_view, output_tensor_view);
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    HCTR_LIB_THROW(cudaMemcpy(h_outs.data(), output_tensor_3d.data(), output_tensor_3d.num_bytes(),
                              cudaMemcpyDeviceToHost));

    int64_t match_count = 0;
    for (int64_t z = 0; z < output_shape.size(0); z++) {
      for (int64_t y = 0; y < output_shape.size(1); y++) {
        for (int64_t x = 0; x < output_shape.size(2); x++) {
          int64_t input_index = (z + OFFSET_Z) * input_tensor_view.stride(0) +
                                (y + OFFSET_Y) * input_tensor_view.stride(1) +
                                (x + OFFSET_X) * input_tensor_view.stride(2);
          int64_t output_index = (z)*output_tensor_view.stride(0) +
                                 (y)*output_tensor_view.stride(1) +
                                 (x)*output_tensor_view.stride(2);
          if (h_outs[output_index] == h_refs[input_index]) {
            match_count++;
          } else {
            HCTR_LOG_S(DEBUG, ROOT)
                << "output[" << z << "][" << y << "][" << x << "] != input[" << z + OFFSET_Z << "]["
                << y + OFFSET_Y << "][" << x + OFFSET_X << "]: " << h_outs[output_index] << " vs. "
                << h_refs[input_index] << std::endl;
          }
        }
      }
    }
    HCTR_LOG_S(DEBUG, ROOT) << "match_count(" << match_count << ") vs. output count("
                            << output_shape.size() << ")" << std::endl;
    EXPECT_TRUE(match_count == output_shape.size());
  }

  // 2D TensorView Test
  {
    BufferParams buffer_params;
    buffer_params.channel = "2D_TENSOR_VIEW_TEST";

    int* input_data = nullptr;
    HCTR_LIB_THROW(cudaMalloc(&input_data, INPUT_BYTES));
    HCTR_LIB_THROW(cudaMemcpy(input_data, h_ins.data(), INPUT_BYTES, cudaMemcpyHostToDevice));

    int* output_data = nullptr;
    HCTR_LIB_THROW(cudaMalloc(&output_data, OUTPUT_BYTES));

    auto input_tensor_params = TensorParams()
                                   .shape({INPUT_SIZE_Y, INPUT_SIZE_X})
                                   .buffer_params(buffer_params)
                                   .data_type(ScalarType::Int32);
    auto input_tensor_2d = Tensor(input_tensor_params);

    auto output_tensor_2d = Tensor(input_tensor_params.shape({{OUTPUT_SIZE_Y, OUTPUT_SIZE_X}}));

    HCTR_LIB_THROW(cudaMemcpy(input_tensor_2d.data(), h_ins.data(), input_tensor_2d.num_bytes(),
                              cudaMemcpyHostToDevice));

    auto input_tensor_view = input_tensor_2d.view<int, Dims - 1>({OFFSET_Y, OFFSET_X});
    auto output_tensor_view = output_tensor_2d.view<int, Dims - 1>();

    EXPECT_FALSE(input_tensor_view.size(0) * input_tensor_view.size(1) ==
                 input_tensor_2d.num_elements());
    EXPECT_TRUE(output_tensor_view.size(0) * output_tensor_view.size(1) ==
                output_tensor_2d.num_elements());

    EXPECT_FALSE(input_tensor_2d.data() == &input_tensor_view[0][0]);
    EXPECT_TRUE(input_tensor_2d.data<int>() + (OFFSET_Y * INPUT_SIZE_X + OFFSET_X) ==
                &input_tensor_view[0][0]);
    EXPECT_TRUE(output_tensor_2d.data() == &output_tensor_view[0][0]);

    auto output_shape = output_tensor_2d.shape();
    dim3 block(32, 32);
    dim3 grid((output_shape.size(1) + block.x - 1) / block.x,
              (output_shape.size(0) + block.y - 1) / block.y);
    copy_kernel<int><<<grid, block>>>(input_tensor_view, output_tensor_view);
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    HCTR_LIB_THROW(cudaMemcpy(h_outs.data(), output_tensor_2d.data(), output_tensor_2d.num_bytes(),
                              cudaMemcpyDeviceToHost));

    int64_t match_count = 0;
    for (int64_t y = 0; y < output_shape.size(0); y++) {
      for (int64_t x = 0; x < output_shape.size(1); x++) {
        int64_t input_index = (y + OFFSET_Y) * input_tensor_view.stride(0) +
                              (x + OFFSET_X) * input_tensor_view.stride(1);
        int64_t output_index = (y)*output_tensor_view.stride(0) + (x)*output_tensor_view.stride(1);
        if (h_outs[output_index] == h_refs[input_index]) {
          match_count++;
        } else {
          HCTR_LOG_S(DEBUG, ROOT) << "output[" << y << "][" << x << "] != input[" << y + OFFSET_Y
                                  << "][" << x + OFFSET_X << "]: " << h_outs[output_index]
                                  << " vs. " << h_refs[input_index] << std::endl;
        }
      }
    }
    HCTR_LOG_S(DEBUG, ROOT) << "match_count(" << match_count << ") vs. output count("
                            << output_shape.size() << ")" << std::endl;
    EXPECT_TRUE(match_count == output_shape.size());

    HCTR_LIB_THROW(cudaFree(input_data));
    HCTR_LIB_THROW(cudaFree(output_data));
  }
}

}  // namespace

TEST(test_core23, tensor_view_3d_test) { tensor_view_test_impl(); }
