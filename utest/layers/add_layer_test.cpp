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

#include "HugeCTR/include/layers/add_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-5f;

template<typename T>
void add_cpu(T ** input, 
              T * output, 
              int size, 
              int num) {
  for (int i = 0; i < size; i++) {
    T tmp = 0;
    for (int j = 0; j < num; j++) {
      tmp += input[j][i];
    }
    output[i] = tmp;
  }
}

template<typename T>
void add_dgrad_cpu(const T * top_grad,
                  T ** dgrad,
                  int size,
                  int num) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < num; j++) {
      dgrad[j][i] = top_grad[i];
    }
  }
}

void add_test(int batch_size, int slot_num, int embedding_vec_size, int num) {
  int dev_id = 0;
  std::shared_ptr<GeneralBuffer<float>> in_out_buf(new GeneralBuffer<float>());

  vector<int> dims_in = {batch_size, slot_num, embedding_vec_size};
  vector<int> dims_out = {batch_size, slot_num, embedding_vec_size};
  int size = batch_size * slot_num * embedding_vec_size;

  std::vector<std::shared_ptr<Tensor<float>>> in_tensors;
  for(int i = 0; i < num; i++) {
    in_tensors.emplace_back(new Tensor<float>(dims_in, in_out_buf, TensorFormat_t::HSW));
  }
  std::shared_ptr<Tensor<float>> out_tensor(new Tensor<float>(dims_out, in_out_buf, TensorFormat_t::HSW));
  in_out_buf->init(dev_id);

  std::unique_ptr<float*[]> h_d_ins(new (float *[num]));
  for(int i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i]->get_ptr();
  }
  float** d_ins;
  CK_CUDA_THROW_(cudaMalloc((void**)(&d_ins), num * sizeof(float*)));
  CK_CUDA_THROW_(cudaMemcpy((void*)d_ins, (void*)h_d_ins.get(), num * sizeof(float*), cudaMemcpyHostToDevice));
  float* d_out = out_tensor->get_ptr();

  std::unique_ptr<float *[]> h_ins(new float *[num]);
  for(int i = 0; i < num; i++) {
    h_ins[i] = new float[size];
  }
  std::unique_ptr<float[]> h_out(new float[size]);
  std::unique_ptr<float[]> h_cpu_out(new float[size]);
  std::unique_ptr<float *[]> h_gpu_dgrads(new float *[num]);
  for(int i = 0; i < num; i++) {
    h_gpu_dgrads[i] = new float[size];
  }

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  AddLayer add_layer(in_tensors, out_tensor, dev_id);

  // fprop
  for(int i = 0; i < num; i++) {
    for (int j = 0; j < size; j++) {
      h_ins[i][j] = simulator.get_num();
    }
    cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(float), cudaMemcpyHostToDevice);
  }  

  add_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

  add_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<float>(h_out.get(), h_cpu_out.get(), size, eps));

  // bprop
  for(int i = 0; i < num; i++) {
    for (int j = 0; j < size; j++) {
      h_ins[i][j] = simulator.get_num();
    }
    cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(float), cudaMemcpyHostToDevice);
  }
  for(int i = 0; i < size; i++) {
    h_out[i] = simulator.get_num(); // top_grad
  }
  cudaMemcpy(d_out, h_out.get(), size * sizeof(float), cudaMemcpyHostToDevice);
  add_layer.bprop(cudaStreamDefault); // compute wgrad and dgrad
  for(int i = 0; i < num; i++) {
    cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(float), cudaMemcpyDeviceToHost);
  }

  add_dgrad_cpu(h_out.get(), h_ins.get(), size, num);
  for(int i = 0; i < num; i++) {
    ASSERT_TRUE(test::compare_array_approx<float>(h_ins[i], h_gpu_dgrads[i], size, eps)); // compare dgrad
  }
}

}  // namespace

TEST(add_layer, fprop_and_bprop) {
  // add_test(1, 1, 32, 2);
  // add_test(4096, 10, 64, 3);
  add_test(40960, 1, 1, 3);
}
