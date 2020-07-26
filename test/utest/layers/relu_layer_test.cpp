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

#include "HugeCTR/include/layers/relu_layer.hpp"

#include "HugeCTR/include/data_parser.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>

using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-6;

template <typename T>
void relu_cpu(T* top, const T* bottom, int len);

template <>
void relu_cpu<float>(float* top, const float* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] < 0) {
      top[i] = 0.0f;
    } else {
      top[i] = bottom[i];
    }
  }
}

template <>
void relu_cpu<__half>(__half* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > 0) {
      top[i] = bottom[i];
    } else {
      top[i] = __float2half(0.0f);
    }
  }
}

template <typename T>
void relu_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len);

template <>
void relu_bprop_cpu<float>(float* d_bottom, const float* d_top, const float* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] < 0) {
      d_bottom[i] = 0.f;
    } else {
      d_bottom[i] = d_top[i];
    }
  }
}

template <>
void relu_bprop_cpu<__half>(__half* d_bottom, const __half* d_top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > 0) {
      d_bottom[i] = d_top[i];
    } else {
      d_bottom[i] = __float2half(0.0f);
    }
  }
}

template <typename T>
void relu_test(size_t dim0, size_t dim1) {
  shared_ptr<GeneralBuffer<T>> buf(new GeneralBuffer<T>());
  vector<size_t> dims = {dim0, dim1};
  shared_ptr<Tensor<T>> bottom_tensor(new Tensor<T>(dims, buf));
  shared_ptr<Tensor<T>> top_tensor(new Tensor<T>(dims, buf));
  buf->init(0);

  const size_t len = dim0 * dim1;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> h_top(new T[len]);
  std::unique_ptr<T[]> d2h_top(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad(new T[len]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[len]);

  GaussianDataSimulator<float> simulator(0.0, 1.0, -2.0, 2.0);
  for (size_t i = 0; i < len; ++i) {
    h_bottom[i] = simulator.get_num();
  }

  // fprop

  ReluLayer<T> relu_layer(bottom_tensor, top_tensor, 0);
  cudaMemcpy(bottom_tensor->get_ptr(), h_bottom.get(), len * sizeof(T), cudaMemcpyHostToDevice);
  relu_layer.fprop(cudaStreamDefault);
  cudaMemcpy(d2h_top.get(), top_tensor->get_ptr(), len * sizeof(T), cudaMemcpyDeviceToHost);

  relu_cpu<T>(h_top.get(), h_bottom.get(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), len, eps));

  // bprop
  for (size_t i = 0; i < len; ++i) {
    h_top[i] = simulator.get_num();
  }
  cudaMemcpy(top_tensor->get_ptr(), h_top.get(), len * sizeof(T), cudaMemcpyHostToDevice);
  relu_layer.bprop(cudaStreamDefault);
  cudaMemcpy(d2h_bottom_grad.get(), bottom_tensor->get_ptr(), len * sizeof(T),
             cudaMemcpyDeviceToHost);

  relu_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_bottom.get(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, eps));
}

}  // namespace

TEST(relu_layer, fp32_fprop_and_bprop) {
  relu_test<float>(10, 20);
  relu_test<float>(10, 500);
  relu_test<float>(512, 1024 * 2);
}

TEST(relu_layer, fp16_fprop_and_bprop) {
  relu_test<__half>(10, 20);
  relu_test<__half>(10, 500);
  relu_test<__half>(512, 1024 * 2);
}
