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


#include "HugeCTR/include/layers/slice_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <math.h>
#include <memory>
#include <set>
#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5;

void slice_layer_test(int height, int width, std::vector<std::pair<int,int>> ranges) {
  std::shared_ptr<GeneralBuffer<float>> buff(new GeneralBuffer<float>());
  std::vector<int> in_dims = {height, width};
  TensorFormat_t in_format = TensorFormat_t::HW;
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(in_dims, buff, in_format));

  GaussianDataSimulator<float> data_sim(0.0, 1.0, -10.0, 10.0);
  std::vector<float> h_in(in_tensor->get_num_elements(), 0.0);
  for (unsigned int i = 0; i < h_in.size(); i++) {
    h_in[i] = data_sim.get_num();
  }

  Tensors<float> out_tensors;
  SliceLayer slice_layer(in_tensor, out_tensors, buff, ranges, 0);

  int n_outs = out_tensors.size();

  buff->init(0);

  // fprop
  std::vector<std::vector<float>> h_refs;
  for(int i = 0; i < n_outs; i++) {
    std::vector<float> h_ref(out_tensors[i]->get_num_elements(), 0.0);
    h_refs.push_back(h_ref);
  }

  int i = 0;
  for(auto& range : ranges) {
    int out_width = range.second - range.first;
    for(int r = 0; r < height; r++) {
      for(int c = range.first; c < range.second; c++) {
        int in_idx = r * width + c;
        int out_idx = r * out_width + c - range.first;
        h_refs[i][out_idx] = h_in[in_idx];
      }
    }
    i++;
  }

  float* d_in = in_tensor->get_ptr();
  cudaMemcpy(d_in, &h_in.front(), in_tensor->get_size(), cudaMemcpyHostToDevice);

  slice_layer.fprop(cudaStreamDefault);

  for(int i = 0; i < n_outs; i++) {
    std::vector<float> h_out(out_tensors[i]->get_num_elements(), 0.0);
    float* d_out = out_tensors[i]->get_ptr();
    cudaMemcpy(&h_out.front(), d_out, out_tensors[i]->get_size(), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(
        test::compare_array_approx<float>(&h_out.front(), &h_refs[i].front(), h_out.size(), eps));
  }

  // bprop
  slice_layer.bprop(cudaStreamDefault);

  for (unsigned int i = 0; i < h_in.size(); i++) {
    h_in[i] = 0.0f;
  }
  i = 0;
  for(auto& range : ranges) {
    int out_width = range.second - range.first;
    for(int r = 0; r < height; r++) {
      for(int c = range.first; c < range.second; c++) {
        int in_idx = r * width + c;
        int out_idx = r * out_width + c - range.first;
        h_in[in_idx] += h_refs[i][out_idx];
      }
    }
    i++;
  }
  std::vector<float> h_out(in_tensor->get_num_elements(), 0.0);
  cudaMemcpy(&h_out.front(), d_in, in_tensor->get_size(), cudaMemcpyDeviceToHost);
  ASSERT_TRUE(
      test::compare_array_approx<float>(&h_out.front(), &h_in.front(), h_out.size(), eps));
}

}  // namespace

TEST(slice_layer, 64x128_0_48_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,48));
  ranges.push_back(std::make_pair(32,64));
  slice_layer_test(64, 128, ranges);
}

TEST(slice_layer, 64x128_0_32_32_64) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,32));
  ranges.push_back(std::make_pair(32,64));
  slice_layer_test(64, 128, ranges);
}

TEST(slice_layer, 64x100_0_40_50_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,40));
  ranges.push_back(std::make_pair(50,90));
  slice_layer_test(64, 100, ranges);
}

TEST(slice_layer, 64x100_0_50_40_90) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,50));
  ranges.push_back(std::make_pair(40,90));
  slice_layer_test(64, 100, ranges);
}

TEST(slice_layer, 64x256_0_50_40_90_80_130) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,50));
  ranges.push_back(std::make_pair(40,90));
  ranges.push_back(std::make_pair(80,130));
  slice_layer_test(64, 256, ranges);
}

TEST(slice_layer, 64x256_0_32_64_80_96_128_128_160_192_256) {
  std::vector<std::pair<int, int>> ranges;
  ranges.push_back(std::make_pair(0,32));
  ranges.push_back(std::make_pair(64,80));
  ranges.push_back(std::make_pair(96,128));
  ranges.push_back(std::make_pair(128,160));
  ranges.push_back(std::make_pair(192,256));
  slice_layer_test(64, 256, ranges);
}



