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

#include "HugeCTR/include/layers/fm_order2_layer.hpp"
#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

namespace {

const float eps = 1e-5f;

void fm_order2_fprop_cpu(const float* in, float* out, int batch_size, int slot_num,
                         int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      float square_sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        float input = in[index];
        sum += input;
        square_sum += input * input;
      }
      float sum_square = sum * sum;
      out[i * emb_vec_size + j] = 0.5f * (sum_square - square_sum);
    }
  }
}

void fm_order2_bprop_cpu(const float* in, const float* top_grad, float* dgrad, int batch_size,
                         int slot_num, int emb_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < emb_vec_size; j++) {
      float sum = 0.0f;
      int offset = i * slot_num * emb_vec_size + j;
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        sum += in[index];
      }
      for (int k = 0; k < slot_num; k++) {
        int index = offset + k * emb_vec_size;
        dgrad[index] = top_grad[i * emb_vec_size + j] * (sum - in[index]);
      }
    }
  }
}

void fm_order2_test(size_t batch_size, size_t slot_num, size_t emb_vec_size) {
  std::shared_ptr<GeneralBuffer<float>> buf(new GeneralBuffer<float>);
  std::vector<size_t> in_dims = {batch_size, slot_num * emb_vec_size};
  std::shared_ptr<Tensor<float>> in_tensor(new Tensor<float>(in_dims, buf, TensorFormat_t::HW));
  std::vector<size_t> out_dims = {batch_size, emb_vec_size};
  std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(out_dims, buf, TensorFormat_t::HW));
  buf->init(0);

  float* d_in = in_tensor->get_ptr();
  float* d_out = out_tensor->get_ptr();

  const size_t in_len = batch_size * slot_num * emb_vec_size;
  const size_t out_len = batch_size * emb_vec_size;
  std::unique_ptr<float[]> h_in(new float[in_len]);
  std::unique_ptr<float[]> h_out(new float[out_len]);
  std::unique_ptr<float[]> h_expected(new float[out_len]);
  std::unique_ptr<float[]> h_expected_dgrad(new float[in_len]);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  FmOrder2Layer fm_order2_layer(in_tensor, out_tensor, 0);

  for (size_t i = 0; i < in_len; i++) {
    h_in[i] = simulator.get_num();
  }

  cudaMemcpy(d_in, h_in.get(), in_len * sizeof(float), cudaMemcpyHostToDevice);
  fm_order2_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost);

  fm_order2_fprop_cpu(h_in.get(), h_expected.get(), batch_size, slot_num, emb_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_expected.get(), out_len, eps));

  for (size_t i = 0; i < in_len; i++) {
    h_in[i] = simulator.get_num();
    h_expected_dgrad[i] = h_in[i];
  }
  for (size_t i = 0; i < out_len; i++) {
    h_out[i] = simulator.get_num();
  }

  cudaMemcpy(d_in, h_in.get(), in_len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.get(), out_len * sizeof(float), cudaMemcpyHostToDevice);
  fm_order2_layer.bprop(cudaStreamDefault);
  cudaMemcpy(h_in.get(), d_in, in_len * sizeof(float), cudaMemcpyDeviceToHost);

  fm_order2_bprop_cpu(h_expected_dgrad.get(), h_out.get(), h_expected_dgrad.get(), batch_size,
                      slot_num, emb_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_expected_dgrad.get(), in_len, eps));
}

}  // end of namespace

TEST(fm_order2_layer, fprop_and_bprop) {
  fm_order2_test(4, 2, 32);
  fm_order2_test(4096, 10, 64);
}
