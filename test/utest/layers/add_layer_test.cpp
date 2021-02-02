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

#include <gtest/gtest.h>
#include <utest/test_utils.h>

#include <vector>

#include "HugeCTR/include/utils.hpp"

using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
struct Eps {
  static T value();
};

template <>
struct Eps<float> {
  static constexpr float value() { return 1e-6f; }
};

template <>
struct Eps<__half> {
  static __half value() { return __float2half(1e-2f); }
};

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
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  vector<size_t> dims_in = {batch_size, slot_num, embedding_vec_size};
  vector<size_t> dims_out = {batch_size, slot_num, embedding_vec_size};
  size_t size = batch_size * slot_num * embedding_vec_size;

  Tensors2<T> in_tensors;
  for (size_t i = 0; i < num; i++) {
    Tensor2<T> in_tensor;
    buff->reserve(dims_in, &in_tensor);
    in_tensors.push_back(in_tensor);
  }
  Tensor2<T> out_tensor;
  buff->reserve(dims_out, &out_tensor);

  AddLayer<T> add_layer(in_tensors, out_tensor, buff, test::get_default_gpu());

  buff->allocate();
  add_layer.initialize();

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }
  T *d_out = out_tensor.get_ptr();

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

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // fprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  add_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));

  add_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), size, Eps<T>::value()));

  // bprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulator.fill(h_out.get(), size);
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), size * sizeof(T), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  add_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  for (size_t i = 0; i < num; i++) {
    CK_CUDA_THROW_(
        cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  add_dgrad_cpu(h_out.get(), h_ins.get(), size, num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(test::compare_array_approx<T>(h_ins[i], h_gpu_dgrads[i], size,
                                              Eps<T>::value()));  // compare dgrad
  }
}

}  // namespace

TEST(add_layer, fp32_40960x1x1) { add_test<float>(40960, 1, 1, 3); }
TEST(add_layer, fp16_40960x1x1) { add_test<__half>(40960, 1, 1, 3); }
TEST(add_layer, fp32_40960x4x3) { add_test<float>(40960, 4, 3, 3); }
TEST(add_layer, fp16_40960x4x3) { add_test<__half>(40960, 4, 3, 3); }
TEST(add_layer, fp32_4096x4x256) { add_test<float>(4096, 4, 256, 3); }
TEST(add_layer, fp16_4096x4x256) { add_test<__half>(4096, 4, 256, 3); }
