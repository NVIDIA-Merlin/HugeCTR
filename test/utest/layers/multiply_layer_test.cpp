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

#include "HugeCTR/include/layers/multiply_layer.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-3f;

template <typename T>
void multiply_cpu(const T* input, const T* weight, T* output, int batch_size, int slot_num,
                  int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      for (int k = 0; k < embedding_vec_size; k++) {
        output[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] =
            input[i * slot_num + j] * weight[j * embedding_vec_size + k];
      }
    }
  }
}

template <typename T>
void multiply_wgrad_cpu(const T* top_grad, const T* input, T* wgrad, int batch_size, int slot_num,
                        int embedding_vec_size) {
  int len_w = slot_num * embedding_vec_size;
  for (int i = 0; i < len_w; i++) {
    double tmp = 0.0;
    for (int j = 0; j < batch_size; j++) {
      tmp += (double)input[j * slot_num + i / embedding_vec_size] * (double)top_grad[j * len_w + i];
    }
    wgrad[i] = (T)tmp;
  }
}

template <typename T>
void multiply_dgrad_cpu(const T* top_grad, const T* weight, T* dgrad, int batch_size, int slot_num,
                        int embedding_vec_size) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < slot_num; j++) {
      T tmp = 0;
      for (int k = 0; k < embedding_vec_size; k++) {
        tmp += top_grad[i * slot_num * embedding_vec_size + j * embedding_vec_size + k] *
               weight[j * embedding_vec_size + k];
      }
      dgrad[i * slot_num + j] = tmp;
    }
  }
}

void multiply_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  std::shared_ptr<BufferBlock2<float>> weight_buff = buff->create_block<float>();
  std::shared_ptr<BufferBlock2<float>> wgrad_buff = buff->create_block<float>();

  vector<size_t> in_dims = {batch_size, slot_num};
  vector<size_t> weight_dims = {slot_num, embedding_vec_size};

  Tensor2<float> in_tensor;
  buff->reserve(in_dims, &in_tensor);
  Tensor2<float> out_tensor;

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  MultiplyLayer multiply_layer(weight_buff, wgrad_buff, buff, in_tensor, out_tensor, weight_dims,
                               test::get_default_gpu());

  buff->allocate();
  multiply_layer.initialize();

  Tensor2<float> weight = weight_buff->as_tensor();
  Tensor2<float> wgrad = wgrad_buff->as_tensor();

  float* d_weight = weight.get_ptr();
  float* d_wgrad = wgrad.get_ptr();

  const size_t len_in = batch_size * slot_num;
  const size_t len_out = batch_size * slot_num * embedding_vec_size;
  const size_t len_w = slot_num * embedding_vec_size;
  float* d_in = in_tensor.get_ptr();
  float* d_out = out_tensor.get_ptr();
  std::unique_ptr<float[]> h_in(new float[len_in]);
  std::unique_ptr<float[]> h_out(new float[len_out]);
  std::unique_ptr<float[]> h_weight(new float[len_w]);
  std::unique_ptr<float[]> h_wgrad(new float[len_w]);
  std::unique_ptr<float[]> h_expected(new float[len_out]);
  std::unique_ptr<float[]> h_expected_wgrad(new float[len_w]);

  // fprop
  simulator.fill(h_in.get(), len_in);
  simulator.fill(h_weight.get(), len_w);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len_in * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(
      cudaMemcpy(d_weight, h_weight.get(), len_w * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  multiply_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, len_out * sizeof(float), cudaMemcpyDeviceToHost));

  multiply_cpu(h_in.get(), h_weight.get(), h_expected.get(), batch_size, slot_num,
               embedding_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_expected.get(), len_out, eps));

  // bprop
  simulator.fill(h_in.get(), len_in);
  for (size_t i = 0; i < len_in; ++i) {
    h_expected[i] = h_in[i];
  }
  simulator.fill(h_out.get(), len_out);
  simulator.fill(h_weight.get(), len_w);
  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), len_in * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), len_out * sizeof(float), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(
      cudaMemcpy(d_weight, h_weight.get(), len_w * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  multiply_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(
      cudaMemcpy(h_wgrad.get(), d_wgrad, len_w * sizeof(float), cudaMemcpyDeviceToHost));  // wgrad
  CK_CUDA_THROW_(
      cudaMemcpy(h_in.get(), d_in, len_in * sizeof(float), cudaMemcpyDeviceToHost));  // dgrad

  multiply_wgrad_cpu(h_out.get(), h_expected.get(), h_expected_wgrad.get(), batch_size, slot_num,
                     embedding_vec_size);
  // TODO: because of the accumulated error, comparing absolute error can not pass when esp<1e-3
  ASSERT_TRUE(test::compare_array_approx<float>(h_wgrad.get(), h_expected_wgrad.get(), len_w,
                                                eps));  // compare wgrad

  // CAUSION: dgrad computation will modify the "input", so it must be put after wgrad computation
  multiply_dgrad_cpu(h_out.get(), h_weight.get(), h_expected.get(), batch_size, slot_num,
                     embedding_vec_size);
  ASSERT_TRUE(test::compare_array_approx<float>(h_in.get(), h_expected.get(), len_in,
                                                eps));  // compare dgrad
}

}  // namespace

TEST(multiply_layer, fp32_40960x10x128) { multiply_test(40960, 10, 128); }
