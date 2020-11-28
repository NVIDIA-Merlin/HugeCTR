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

#include "HugeCTR/include/layers/dot_product_layer.hpp"
#include <vector>
#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5f;

template <typename T>
void dot_product_cpu(T **input, T *output, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    T tmp = 1;
    for (size_t j = 0; j < num; j++) {
      tmp *= input[j][i];
    }
    output[i] = tmp;
  }
}

template <typename T>
void dot_product_dgrad_cpu(const T *top_grad, T **dgrad, const T *fprop_output, size_t size,
                           size_t num) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      if (0 == fprop_output[i]) {
        dgrad[j][i] = 0;
      } else {
        T d_input = dgrad[j][i];
        dgrad[j][i] = top_grad[i] * ((float)fprop_output[i] / d_input);
      }
    }
  }
}

void dot_product_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size, size_t num) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  vector<size_t> dims_in = {batch_size, slot_num, embedding_vec_size};
  vector<size_t> dims_out = {batch_size, slot_num, embedding_vec_size};
  size_t size = batch_size * slot_num * embedding_vec_size;

  Tensors2<float> in_tensors;
  for (size_t i = 0; i < num; i++) {
    Tensor2<float> in_tensor;
    buff->reserve(dims_in, &in_tensor);
    in_tensors.push_back(in_tensor);
  }
  Tensor2<float> out_tensor;
  buff->reserve(dims_out, &out_tensor);

  DotProductLayer dot_product_layer(in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();
  dot_product_layer.initialize();

  std::unique_ptr<float *[]> h_d_ins(new float *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }
  float **d_ins;
  CK_CUDA_THROW_(cudaMalloc((void **)(&d_ins), num * sizeof(float *)));
  CK_CUDA_THROW_(cudaMemcpy((void *)d_ins, (void *)h_d_ins.get(), num * sizeof(float *),
                            cudaMemcpyHostToDevice));
  float *d_out = out_tensor.get_ptr();

  std::unique_ptr<float *[]> h_ins(new float *[num]);
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new float[size];
  }
  std::unique_ptr<float[]> h_out(new float[size]);
  std::unique_ptr<float[]> fprop_output(new float[size]);
  std::unique_ptr<float[]> h_cpu_out(new float[size]);
  std::unique_ptr<float *[]> h_gpu_dgrads(new float *[num]);
  for (size_t i = 0; i < num; i++) {
    h_gpu_dgrads[i] = new float[size];
  }

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(float), cudaMemcpyHostToDevice));
  }

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  dot_product_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(
      cudaMemcpy(fprop_output.get(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost));

  dot_product_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_cpu_out.get(), size, eps));

  // bprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(float), cudaMemcpyHostToDevice));
  }
  simulator.fill(h_out.get(), size);
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), size * sizeof(float), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  dot_product_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  for (size_t i = 0; i < num; i++) {
    CK_CUDA_THROW_(
        cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(float), cudaMemcpyDeviceToHost));
  }

  dot_product_dgrad_cpu(h_out.get(), h_ins.get(), fprop_output.get(), size, num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(
        test::compare_array_approx<float>(h_ins[i], h_gpu_dgrads[i], size, eps));  // compare dgrad
  }
}

}  // namespace

TEST(dot_product_layer, fp32_40960x1x1_3) { dot_product_test(40960, 1, 1, 3); }
