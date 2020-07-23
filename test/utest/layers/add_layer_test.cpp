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

#include "HugeCTR/include/layers/add_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-2f;

template <typename T>
void add_cpu(T **input, T *output, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += input[j][i];
    }
    output[i] = tmp;
  }
}

template <>
void add_cpu(__half **input, __half *output, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += __half2float(input[j][i]);
    }
    output[i] = __float2half(tmp);
  }
}

template <typename T>
void add_dgrad_cpu(const T *top_grad, T **dgrad, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      dgrad[j][i] = top_grad[i];
    }
  }
}

template <typename T>
void add_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size, size_t num) {
  size_t dev_id = 0;
  std::shared_ptr<GeneralBuffer<T>> in_out_buf(new GeneralBuffer<T>());

  vector<size_t> dims_in = {batch_size, slot_num, embedding_vec_size};
  vector<size_t> dims_out = {batch_size, slot_num, embedding_vec_size};
  size_t size = batch_size * slot_num * embedding_vec_size;

  std::vector<std::shared_ptr<Tensor<T>>> in_tensors;
  for (size_t i = 0; i < num; i++) {
    in_tensors.emplace_back(new Tensor<T>(dims_in, in_out_buf, TensorFormat_t::HSW));
  }
  std::shared_ptr<Tensor<T>> out_tensor(
      new Tensor<T>(dims_out, in_out_buf, TensorFormat_t::HSW));
  in_out_buf->init(dev_id);

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i]->get_ptr();
  }
  T **d_ins;
  CK_CUDA_THROW_(cudaMalloc((void **)(&d_ins), num * sizeof(T *)));
  CK_CUDA_THROW_(cudaMemcpy((void *)d_ins, (void *)h_d_ins.get(), num * sizeof(T *),
                            cudaMemcpyHostToDevice));
  T *d_out = out_tensor->get_ptr();

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new T[size];
  }
  std::unique_ptr<T[]> h_out(new T[size]);
  std::unique_ptr<T[]> h_cpu_out(new T[size]);
  std::unique_ptr<T *[]> h_gpu_dgrads(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_gpu_dgrads[i] = new T[size];
  }

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  AddLayer<T> add_layer(in_tensors, out_tensor, dev_id);

  // fprop
  for (size_t i = 0; i < num; i++) {
    for (size_t j = 0; j < size; j++) {
      h_ins[i][j] = TypeConvert<T>::convert(simulator.get_num());
    }
    cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice);
  }

  add_layer.fprop(cudaStreamDefault);
  cudaMemcpy(h_out.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost);

  add_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), size, eps));

  // bprop
  for (size_t i = 0; i < num; i++) {
    for (size_t j = 0; j < size; j++) {
      h_ins[i][j] = TypeConvert<T>::convert(simulator.get_num());
    }
    cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice);
  }
  for (size_t i = 0; i < size; i++) {
    h_out[i] = TypeConvert<T>::convert(simulator.get_num());  // top_grad
  }
  cudaMemcpy(d_out, h_out.get(), size * sizeof(T), cudaMemcpyHostToDevice);
  add_layer.bprop(cudaStreamDefault);  // compute wgrad and dgrad
  for (size_t i = 0; i < num; i++) {
    cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost);
  }

  add_dgrad_cpu(h_out.get(), h_ins.get(), size, num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(
        test::compare_array_approx<T>(h_ins[i], h_gpu_dgrads[i], size, eps));  // compare dgrad
  }
}

}  // namespace

TEST(add_layer, fp32_fprop_and_bprop) {
  // add_test(1, 1, 32, 2);
  // add_test(4096, 10, 64, 3);
  add_test<float>(40960, 1, 1, 3);
}

TEST(add_layer, fp16_fprop_and_bprop) {
  add_test<__half>(40960, 1, 1, 3);
}