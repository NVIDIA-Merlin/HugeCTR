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

#include "HugeCTR/include/layers/sigmoid_layer.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda_fp16.h>
using namespace std;
using namespace HugeCTR;

namespace {

const float eps = 1e-2;

template <typename T>
void sigmoid_cpu(T* top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = T(1.) / ( T(1.) + exp(-bottom[i]) );
  }
}

template<>
void sigmoid_cpu(__half* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(1.0 / ( 1.0 + exp(-__half2float(bottom[i])) ) );
  }
}

template <typename T>
void sigmoid_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    T y = T(1.) / ( T(1.) + exp(-bottom[i]) );
    d_bottom[i] = d_top[i] * y * (T(1.)-y);
  }
}

template<>
void sigmoid_bprop_cpu(__half* d_bottom, const __half* d_top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
      float y = 1.0 / ( 1.0 + exp(-__half2float(bottom[i])) );
      d_bottom[i] = __float2half(__half2float(d_top[i]) * y * (1.0 - y));
  }
}

template <typename T>
void sigmoid_test(size_t dim0, size_t dim1) {
  shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  vector<size_t> dims = {dim0, dim1};

  Tensor2<T> bottom_tensor;
  buf->reserve(dims, &bottom_tensor);
  Tensor2<T> top_tensor;
  buf->reserve(dims, &top_tensor);

  SigmoidLayer<T> sigmoid_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  buf->allocate();
  sigmoid_layer.initialize();

  const size_t len = dim0 * dim1;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> h_top(new T[len]);
  std::unique_ptr<T[]> d2h_top(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad(new T[len]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[len]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  simulator.fill(h_bottom.get(), len);

  // fprop

  CK_CUDA_THROW_(
      cudaMemcpy(bottom_tensor.get_ptr(), h_bottom.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  sigmoid_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(
      cudaMemcpy(d2h_top.get(), top_tensor.get_ptr(), len * sizeof(T), cudaMemcpyDeviceToHost));

  sigmoid_cpu<T>(h_top.get(), h_bottom.get(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), len, eps));

  // bprop
  simulator.fill(h_top.get(), len);

  CK_CUDA_THROW_(
      cudaMemcpy(top_tensor.get_ptr(), h_top.get(), len * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  sigmoid_layer.bprop();
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(d2h_bottom_grad.get(), bottom_tensor.get_ptr(), len * sizeof(T),
                            cudaMemcpyDeviceToHost));

  sigmoid_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_bottom.get(), len);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, eps));
}

}  // namespace

TEST(sigmoid_layer, fp32_10x20) { sigmoid_test<float>(10, 20); }
TEST(sigmoid_layer, fp32_10x500) { sigmoid_test<float>(10, 500); }
TEST(sigmoid_layer, fp32_512x2048) { sigmoid_test<float>(512, 1024 * 2); }
TEST(sigmoid_layer, fp16_10x20) { sigmoid_test<__half>(10, 20); }
TEST(sigmoid_layer, fp16_10x500) { sigmoid_test<__half>(10, 500); }
TEST(sigmoid_layer, fp16_512x2048) { sigmoid_test<__half>(512, 1024 * 2); }


