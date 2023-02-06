/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include <layers/softmax_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
using namespace std;

namespace {

const float eps = 1e-4;

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
void sum_grad_softmax(const T* d_top, const T* softmax_out, int embedding_vector_size, int dim0,
                      T* workspace) {
  for (int i = 0; i < dim0; i++) {
    float grad_sum = 0.0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      grad_sum += (float)(d_top[offset + j] * softmax_out[offset + j]);
    }
    workspace[i] = static_cast<T>(grad_sum);
    // printf("CPU grad_sum %d: %f\n", i, workspace[i]);
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
  delete[] workspace;
}

template <typename T>
void softmax_bprop_cpu(T* d_bottom, const T* d_top, const T* softmax_out, int len,
                       int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T* workspace = new T[dim0];

  sum_grad_softmax(d_top, softmax_out, embedding_vector_size, dim0, workspace);
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      d_bottom[index] = softmax_out[index] * (d_top[index] - workspace[i]);
      // d_bottom[index] = workspace[i];
    }
  }
  delete[] workspace;
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
  std::unique_ptr<T[]> h_softmax_out(new T[len]);
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

  softmax_fprop_cpu<T>(h_softmax_out.get(), h_bottom.get(), len, embedding_vector_size);
  HCTR_LIB_THROW(
      cudaMemcpy(top_tensor.get_ptr(), h_top.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(softmax_layer.get_softmax_tensor().get_ptr(), h_softmax_out.get(),
                            len * sizeof(T), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  softmax_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaMemcpy(d2h_bottom_grad.get(), bottom_tensor.get_ptr(), len * sizeof(T),
                            cudaMemcpyDeviceToHost));

  softmax_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_softmax_out.get(), len,
                       embedding_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, eps));
}

}  // namespace

TEST(softmax_layer, fp32_100x100) { softmax_test<float>(100, 100); }
TEST(softmax_layer, fp32_100x128) { softmax_test<float>(100, 128); }
TEST(softmax_layer, fp32_256x384) { softmax_test<float>(256, 384); }
TEST(softmax_layer, fp32_512x512) { softmax_test<float>(512, 512); }
TEST(softmax_layer, fp32_256x1024) { softmax_test<float>(256, 1024); }
TEST(softmax_layer, fp32_1024x512) { softmax_test<float>(1024, 512); }
