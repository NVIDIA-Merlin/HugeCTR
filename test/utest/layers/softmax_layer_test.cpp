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

#include "HugeCTR/include/layers/softmax_layer.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace HugeCTR;

namespace {

const float eps = 1e-6;

template <typename T>
void sum_ex_cpu(T* top, int embedding_vector_size, int dim0, T* workspace) {
  // sum(e^xi) i = [0, embedding_vector_size -1];
  for (int i = 0; i < dim0; i++) {
    workspace[i] = 0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      workspace[i] += top[offset + j];
    }
  }
}

template <typename T>
void ex_cpu(T* top, const T* bottom, int len) {
  // e^xi
  for (int i = 0; i < len; i++) {
    top[i] = expf(bottom[i]);
  }
}

template <typename T>
void softmax_fprop_cpu(T* top, const T* bottom, int len, int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T* workspace = new T[dim0];
  // e^xi
  ex_cpu(top, bottom, len);
  // sum(e^xi) i = [0, embedding_vector_size -1];
  sum_ex_cpu(top, embedding_vector_size, dim0, workspace);
  // softmax : e^xi / sum(e^xi); i = [0, len - 1];
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      top[index] = top[index] / workspace[i];
    }
  }
  delete workspace;
}

template <typename T>
void softmax_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len,
                       int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T* workspace = new T[dim0];
  // e^xi
  ex_cpu(d_bottom, bottom, len);
  // sum(e^xi) i = [0, len - 1];
  sum_ex_cpu(d_bottom, embedding_vector_size, dim0, workspace);
  // softmax derivative :
  // q(x) = e^xi / sum(e^xi);
  // f(x) = q(x) * (1 - q(x) / sum(e^xi));
  // Y = f(x)*d_top i = [0, len - 1];
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      d_bottom[index] = d_bottom[index] / workspace[i] *
                        (1 - d_bottom[index] / pow(workspace[i], 2.0)) * d_top[index];
    }
  }
  delete workspace;
}

template <typename T>
void softmax_test(size_t dim0, size_t embedding_vector_size) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  std::vector<size_t> dims = {dim0, embedding_vector_size};

  Tensor2<T> bottom_tensor;
  buf->reserve(dims, &bottom_tensor);
  Tensor2<T> top_tensor;
  buf->reserve(dims, &top_tensor);

  SoftmaxLayer<T> softmax_layer(bottom_tensor, top_tensor, buf, test::get_default_gpu());

  buf->allocate();
  softmax_layer.initialize();

  const size_t len = dim0 * embedding_vector_size;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> h_top(new T[len]);
  std::unique_ptr<T[]> d2h_top(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad(new T[len]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_bottom.get(), len);

  // fprop
  HCTR_LIB_THROW(
      cudaMemcpy(bottom_tensor.get_ptr(), h_bottom.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  softmax_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(
      cudaMemcpy(d2h_top.get(), top_tensor.get_ptr(), len * sizeof(T), cudaMemcpyDeviceToHost));

  softmax_fprop_cpu<T>(h_top.get(), h_bottom.get(), len, embedding_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), len, eps));

  // bprop
  simulator.fill(h_top.get(), len);

  HCTR_LIB_THROW(
      cudaMemcpy(top_tensor.get_ptr(), h_top.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  softmax_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(d2h_bottom_grad.get(), bottom_tensor.get_ptr(), len * sizeof(T),
                            cudaMemcpyDeviceToHost));

  softmax_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_bottom.get(), len,
                       embedding_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, eps));
}

}  // namespace

TEST(softmax_layer, fp32_10x20) { softmax_test<float>(10, 20); }
TEST(softmax_layer, fp32_100x100) { softmax_test<float>(100, 100); }
TEST(softmax_layer, fp32_100x500) { softmax_test<float>(100, 500); }
TEST(softmax_layer, fp32_512x512) { softmax_test<float>(512, 512); }
TEST(softmax_layer, fp32_512x1048) { softmax_test<float>(512, 1024); }
TEST(softmax_layer, fp32_512x2048) { softmax_test<float>(512, 1024 * 2); }
TEST(softmax_layer, fp32_1048x512) { softmax_test<float>(1024, 512); }
TEST(softmax_layer, fp32_2048x512) { softmax_test<float>(1024 * 2, 512); }
