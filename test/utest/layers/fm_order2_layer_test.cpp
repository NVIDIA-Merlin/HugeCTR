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

#include "HugeCTR/include/layers/fm_order2_layer.hpp"

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
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

  std::vector<size_t> in_dims = {batch_size, slot_num * emb_vec_size};
  Tensor2<float> in_tensor;
  buf->reserve(in_dims, &in_tensor);

  std::vector<size_t> out_dims = {batch_size, emb_vec_size};
  Tensor2<float> out_tensor;
  buf->reserve(out_dims, &out_tensor);

  FmOrder2Layer fm_order2_layer(in_tensor, out_tensor, test::get_default_gpu());

  buf->allocate();
  fm_order2_layer.initialize();

  float* d_in = in_tensor.get_ptr();
  float* d_out = out_tensor.get_ptr();

  const size_t in_len = batch_size * slot_num * emb_vec_size;
  const size_t out_len = batch_size * emb_vec_size;
  std::unique_ptr<float[]> h_in(new float[in_len]);
  std::unique_ptr<float[]> h_out(new float[out_len]);
  std::unique_ptr<float[]> h_expected(new float[out_len]);
  std::unique_ptr<float[]> h_expected_dgrad(new float[in_len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  simulator.fill(h_in.get(), in_len);

  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), in_len * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fm_order2_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, out_len * sizeof(float), cudaMemcpyDeviceToHost));

  fm_order2_fprop_cpu(h_in.get(), h_expected.get(), batch_size, slot_num, emb_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_expected.get(), out_len, eps));

  simulator.fill(h_in.get(), in_len);
  for (size_t i = 0; i < in_len; i++) {
    h_expected_dgrad[i] = h_in[i];
  }
  simulator.fill(h_out.get(), out_len);

  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), in_len * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), out_len * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  fm_order2_layer.bprop();
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_in.get(), d_in, in_len * sizeof(float), cudaMemcpyDeviceToHost));

  fm_order2_bprop_cpu(h_expected_dgrad.get(), h_out.get(), h_expected_dgrad.get(), batch_size,
                      slot_num, emb_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_expected_dgrad.get(), in_len, eps));
}

}  // end of namespace

TEST(fm_order2_layer, fp32_4x2x32) { fm_order2_test(4, 2, 32); }
TEST(fm_order2_layer, fp32_4096x10x64) { fm_order2_test(4096, 10, 64); }
