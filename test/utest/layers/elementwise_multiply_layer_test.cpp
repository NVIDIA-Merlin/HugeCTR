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

#include "HugeCTR/include/layers/elementwise_multiply_layer.hpp"

#include <gtest/gtest.h>
#include <utest/test_utils.h>

#include <vector>

#include "HugeCTR/include/utils.hpp"

using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
T eps();

template <>
constexpr float eps() {
  return 1e-5f;
}

template <>
__half eps() {
  return __float2half(1e-3f);
}

template <typename T>
void elementwise_multiply_cpu(T **input, T *output, size_t size, size_t num) {
  T one = 1.0;

  for (size_t i = 0; i < size; i++) {
    T tmp = one;
    for (size_t j = 0; j < num; j++) {
      tmp = tmp * input[j][i];
    }
    output[i] = tmp;
  }
}

template <typename T>
void elementwise_multiply_dgrad_cpu(const T *top_grad, T **dgrad, const T *fprop_output,
                                    size_t size, size_t num) {
  T zero = 0.0;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      if (0 == fprop_output[i]) {
        dgrad[j][i] = zero;
      } else {
        T d_input = dgrad[j][i];
        dgrad[j][i] = top_grad[i] * T(fprop_output[i] / d_input);
      }
    }
  }
}

template <typename T>
void elementwise_multiply_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                               size_t num) {
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

  ElementwiseMultiplyLayer<T> elementwise_multiply_layer(in_tensors, out_tensor, buff,
                                                         test::get_default_gpu());

  buff->allocate();
  elementwise_multiply_layer.initialize();

  std::unique_ptr<T *[]> h_d_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }
  T **d_ins;
  CK_CUDA_THROW_(cudaMalloc((void **)(&d_ins), num * sizeof(T *)));
  CK_CUDA_THROW_(
      cudaMemcpy((void *)d_ins, (void *)h_d_ins.get(), num * sizeof(T *), cudaMemcpyHostToDevice));
  T *d_out = out_tensor.get_ptr();

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new T[size];
  }
  std::unique_ptr<T[]> h_out(new T[size]);
  std::unique_ptr<T[]> fprop_output(new T[size]);
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
  elementwise_multiply_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
  CK_CUDA_THROW_(cudaMemcpy(fprop_output.get(), d_out, size * sizeof(T), cudaMemcpyDeviceToHost));

  elementwise_multiply_cpu(h_ins.get(), h_cpu_out.get(), size, num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), size, eps<T>()));

  // bprop
  for (size_t i = 0; i < num; i++) {
    simulator.fill(h_ins[i], size);
    CK_CUDA_THROW_(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  simulator.fill(h_out.get(), size);
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), size * sizeof(T), cudaMemcpyHostToDevice));

  CK_CUDA_THROW_(cudaDeviceSynchronize());
  elementwise_multiply_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());

  for (size_t i = 0; i < num; i++) {
    CK_CUDA_THROW_(
        cudaMemcpy(h_gpu_dgrads[i], h_d_ins[i], size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  elementwise_multiply_dgrad_cpu(h_out.get(), h_ins.get(), fprop_output.get(), size, num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_TRUE(
        test::compare_array_approx<T>(h_ins[i], h_gpu_dgrads[i], size, eps<T>()));  // compare dgrad
  }
}

}  // namespace

TEST(elementwise_multiply_layer, fp32_40960x1x1) {
  elementwise_multiply_test<float>(40960, 1, 1, 3);
}
TEST(elementwise_multiply_layer, fp16_40960x1x1) {
  elementwise_multiply_test<__half>(40960, 1, 1, 3);
}
TEST(elementwise_multiply_layer, fp32_40960x4x3) {
  elementwise_multiply_test<float>(40960, 4, 3, 3);
}
TEST(elementwise_multiply_layer, fp16_40960x4x3) {
  elementwise_multiply_test<__half>(40960, 4, 3, 3);
}
TEST(elementwise_multiply_layer, fp32_4096x4x256) {
  elementwise_multiply_test<float>(4096, 4, 256, 3);
}
TEST(elementwise_multiply_layer, fp16_4096x4x256) {
  elementwise_multiply_test<__half>(4096, 4, 256, 3);
}
