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

#include "HugeCTR/include/layers/concat_layer.hpp"
#include <math.h>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5;

template <typename T>
void concat_layer_test(size_t height, std::vector<size_t> widths) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  Tensors2<T> in_tensors;

  test::GaussianDataSimulator data_sim(0.0, 1.0);
  std::vector<std::vector<T>> h_ins;

  int n_ins = widths.size();

  size_t new_width = 0;
  for (int i = 0; i < n_ins; i++) {
    size_t width = widths[i];
    new_width += width;
    std::vector<size_t> in_dims = {height, width};
    Tensor2<T> tensor;
    buff->reserve(in_dims, &tensor);
    in_tensors.push_back(tensor);

    std::vector<T> h_in(height * width, 0.0);
    data_sim.fill(h_in.data(), h_in.size());

    h_ins.push_back(h_in);
  }

  Tensor2<T> out_tensor;
  ConcatLayer<T> concat_layer(in_tensors, in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();
  concat_layer.initialize();

  // fprop
  std::vector<T> h_ref(out_tensor.get_num_elements(), 0.0);
  for (size_t r = 0; r < height; r++) {
    for (size_t c = 0; c < new_width; c++) {
      int out_idx = r * new_width + c;
      int in_no = 0;
      int c2 = c;
      size_t accum_width = 0;
      for (int k = 0; k < n_ins; k++) {
        if (c < accum_width + widths[k]) {
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

  for (int i = 0; i < n_ins; i++) {
    T* d_in = in_tensors[i].get_ptr();
    std::vector<T>& h_in = h_ins[i];
    CK_CUDA_THROW_(
        cudaMemcpy(d_in, &h_in.front(), in_tensors[i].get_size_in_bytes(), cudaMemcpyHostToDevice));
  }

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  concat_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  std::vector<T> h_out(out_tensor.get_num_elements(), 0.0);
  T* d_out = out_tensor.get_ptr();
  CK_CUDA_THROW_(
      cudaMemcpy(&h_out.front(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(&h_out.front(), &h_ref.front(), h_out.size(), eps));

  // bprop
  concat_layer.bprop();
  concat_layer.fprop(true);

  CK_CUDA_THROW_(
      cudaMemcpy(&h_out.front(), d_out, out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost));

  ASSERT_TRUE(test::compare_array_approx<T>(&h_out.front(), &h_ref.front(), h_out.size(), eps));
}

}  // namespace

TEST(concat_layer, fp32_64x32_64x32) { concat_layer_test<float>(64, {32, 32}); }

TEST(concat_layer, fp32_5x32_5x32) { concat_layer_test<float>(5, {32, 32}); }

TEST(concat_layer, fp32_4096x640_4096x1280) { concat_layer_test<float>(4096, {640, 1280}); }

TEST(concat_layer, fp32_64x32_64x64_64x96) { concat_layer_test<float>(64, {32, 64, 96}); }

TEST(concat_layer, fp32_64x32_64x64_64x32_64x128) {
  concat_layer_test<float>(64, {32, 64, 32, 128});
}

TEST(concat_layer, fp32_64x32_64x64_64x32_64x128_64x256) {
  concat_layer_test<float>(64, {32, 64, 32, 128, 256});
}

TEST(concat_layer, fp16_64x32_64x32) { concat_layer_test<__half>(64, {32, 32}); }

TEST(concat_layer, fp16_5x32_5x32) { concat_layer_test<__half>(5, {32, 32}); }

TEST(concat_layer, fp16_4096x640_4096x1280) { concat_layer_test<__half>(4096, {640, 1280}); }

TEST(concat_layer, fp16_64x32_64x64_64x96) { concat_layer_test<__half>(64, {32, 64, 96}); }

TEST(concat_layer, fp16_64x32_64x64_64x32_64x128) {
  concat_layer_test<__half>(64, {32, 64, 32, 128});
}

TEST(concat_layer, fp16_64x32_64x64_64x32_64x128_64x256) {
  concat_layer_test<__half>(64, {32, 64, 32, 128, 256});
}
