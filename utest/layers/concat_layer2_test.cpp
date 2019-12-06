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


#include "HugeCTR/include/layers/concat_layer2.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <math.h>
#include <memory>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5;

void concat_layer2_test(int height, std::vector<int> widths) {
  GeneralBuffer<float> buff;
  TensorFormat_t in_format = TensorFormat_t::HW;
  std::vector<Tensor<float>*> in_tensors;

  GaussianDataSimulator<float> data_sim(0.0, 1.0, -10.0, 10.0);
  std::vector<std::vector<float>> h_ins;

  int n_ins = widths.size();

  int new_width = 0;
  for(int i = 0; i < n_ins; i++) {
    int width = widths[i];
    new_width += width;
    std::vector<int> in_dims = {height, width};
    Tensor<float>* in_tensor = new Tensor<float>(in_dims, buff, in_format);
    in_tensors.push_back(in_tensor);

    std::vector<float> h_in(in_tensor->get_num_elements(), 0.0);
    for (unsigned int i = 0; i < h_in.size(); i++) {
      h_in[i] = data_sim.get_num();
    }
    h_ins.push_back(h_in);
  }

  Tensor<float>* out_tensor = nullptr;
  ConcatLayer2 concat_layer2(in_tensors, &out_tensor, buff, 0);

  buff.init(0);

  // fprop
  std::vector<float> h_ref(out_tensor->get_num_elements(), 0.0);
  for(int r = 0; r < height; r++) {
    for(int c = 0; c < new_width; c++) {
      int out_idx = r * new_width + c;
      int in_no = 0;
      int c2 = c;
      int accum_width = 0;
      for(int k = 0; k < n_ins; k++) {
        if(c < accum_width + widths[k]) {
          in_no = k;
          c2 -= accum_width;
          break;
        }
        accum_width += widths[k];
      }
      int in_idx = r * widths[in_no] + c2;
      h_ref[out_idx] = h_ins[in_no][in_idx];
    }
  }

  for(int i = 0; i < n_ins; i++) {
    float* d_in = in_tensors[i]->get_ptr();
    std::vector<float>& h_in = h_ins[i];
    cudaMemcpy(d_in, &h_in.front(), in_tensors[i]->get_size(), cudaMemcpyHostToDevice);
  }

  concat_layer2.fprop(cudaStreamDefault);

  std::vector<float> h_out(out_tensor->get_num_elements(), 0.0);
  float* d_out = out_tensor->get_ptr();
  cudaMemcpy(&h_out.front(), d_out, out_tensor->get_size(), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(
      test::compare_array_approx<float>(&h_out.front(), &h_ref.front(), h_out.size(), eps));

  // bprop
  concat_layer2.bprop(cudaStreamDefault);
  concat_layer2.fprop(cudaStreamDefault);

  cudaMemcpy(&h_out.front(), d_out, out_tensor->get_size(), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(
      test::compare_array_approx<float>(&h_out.front(), &h_ref.front(), h_out.size(), eps));
}

}  // namespace

TEST(concat_layer2, 64x32_64x32) {
  concat_layer2_test(64, {32, 32});
}

TEST(concat_layer2, 5x32_5x32) {
  concat_layer2_test(5, {32, 32});
}

TEST(concat_layer2, 4096x640_4096x1280) {
  concat_layer2_test(4096, {640, 1280});
}

TEST(concat_layer2, 64x32_64x64_64x96) {
  concat_layer2_test(64, {32, 64, 96});
}

TEST(concat_layer2, 64x32_64x64_64x32_64x128) {
  concat_layer2_test(64, {32, 64, 32, 128});
}

TEST(concat_layer2, 64x32_64x64_64x32_64x128_64x256) {
  concat_layer2_test(64, {32, 64, 32, 128, 256});
}


