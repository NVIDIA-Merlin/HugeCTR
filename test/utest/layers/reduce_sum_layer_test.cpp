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

#include "HugeCTR/include/layers/reduce_sum_layer.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "utest/test_utils.h"

using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
T Eps();

template <>
constexpr float Eps() {
  return 1e-5;
}

template <>
__half Eps() {
  return __float2half(1e-1f);
}

template <typename T>
void reduce_sum_cpu(const T* input, T* output, std::vector<size_t> dims, size_t axis) {
  if (axis == 0) {
    if (dims.size() == 1) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[0] = input[i];
      }
    } else if (dims.size() == 2) {
      for (size_t k = 0; k < dims[1]; k++) {
        output[k] = 0.0f;
        for (size_t i = 0; i < dims[0]; i++) {
          output[k] = output[k] + input[i * dims[1] + k];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[j * dims[2] + k] = 0.0f;
          for (size_t i = 0; i < dims[0]; i++) {
            output[j * dims[2] + k] =
                output[j * dims[2] + k] + input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        output[i] = 0.0f;
        for (size_t j = 0; j < dims[1]; j++) {
          output[i] = output[i] + input[i * dims[1] + j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[2] + k] = 0.0f;
          for (size_t j = 0; j < dims[1]; j++) {
            output[i * dims[2] + k] =
                output[i * dims[2] + k] + input[i * dims[1] * dims[2] + j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        output[i * dims[1] + j] = 0.0f;
        for (size_t k = 0; k < dims[2]; k++) {
          output[i * dims[1] + j] =
              output[i * dims[1] + j] + input[i * dims[1] * dims[2] + j * dims[2] + k];
        }
      }
    }
  }
}

template <typename T>
void reduce_sum_dgrad_cpu(const T* top_grad, T* dgrad, std::vector<size_t> dims, size_t axis) {
  if (axis == 0) {
    if (dims.size() == 2) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t i = 0; i < dims[0]; i++) {
          dgrad[i * dims[1] + j] = top_grad[j];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t i = 0; i < dims[0]; i++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[j * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (dims.size() == 2) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
          dgrad[i * dims[1] + j] = top_grad[i];
        }
      }
    } else if (dims.size() == 3) {
      for (size_t i = 0; i < dims[0]; i++) {
        for (size_t k = 0; k < dims[2]; k++) {
          for (size_t j = 0; j < dims[1]; j++) {
            dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[2] + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t k = 0; k < dims[2]; k++) {
          dgrad[i * dims[1] * dims[2] + j * dims[2] + k] = top_grad[i * dims[1] + j];
        }
      }
    }
  }
}

template <typename T>
void reduce_sum_test(size_t batch_size, size_t slot_num, size_t embedding_vec_size, size_t axis) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  vector<size_t> in_dims = {batch_size, slot_num, embedding_vec_size};  // 3D

  Tensor2<T> in_tensor;
  buff->reserve(in_dims, &in_tensor);
  Tensor2<T> out_tensor;

  test::GaussianDataSimulator simulator(0.0f, 1.0f);
  ReduceSumLayer<T> reduce_sum_layer(in_tensor, out_tensor, buff, axis, test::get_default_gpu());

  buff->allocate();
  reduce_sum_layer.initialize();

  size_t in_size = 1;
  for (auto dim : in_dims) {
    in_size *= dim;
  }
  auto out_dims = out_tensor.get_dimensions();
  size_t out_size = 1;
  for (auto dim : out_dims) {
    out_size *= dim;
  }

  T* d_in = in_tensor.get_ptr();
  T* d_out = out_tensor.get_ptr();
  std::unique_ptr<T[]> h_in(new T[in_size]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T[]> h_gpu_dgrad(new T[in_size]);

  // fprop
  simulator.fill(h_in.get(), in_size);

  CK_CUDA_THROW_(cudaMemcpy(d_in, h_in.get(), in_size * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  reduce_sum_layer.fprop(true);
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(h_out.get(), d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));

  reduce_sum_cpu<T>(h_in.get(), h_cpu_out.get(), in_dims, axis);

  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size, Eps<T>()));

  // bprop
  simulator.fill(h_out.get(), out_size);
  CK_CUDA_THROW_(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  reduce_sum_layer.bprop();  // compute wgrad and dgrad
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaMemcpy(h_gpu_dgrad.get(), d_in, in_size * sizeof(T), cudaMemcpyDeviceToHost));

  reduce_sum_dgrad_cpu<T>(h_out.get(), h_in.get(), in_dims, axis);
  ASSERT_TRUE(test::compare_array_approx<T>(h_in.get(), h_gpu_dgrad.get(), in_size,
                                            Eps<T>()));  // compare dgrad
}

}  // namespace

TEST(reduce_sum_layer, fp32_2x3x4_0) { reduce_sum_test<float>(2, 3, 4, 0); }
TEST(reduce_sum_layer, fp32_2x3x4_1) { reduce_sum_test<float>(2, 3, 4, 1); }
TEST(reduce_sum_layer, fp32_2x3x4_2) { reduce_sum_test<float>(2, 3, 4, 2); }
TEST(reduce_sum_layer, fp32_40960x39x1_1) { reduce_sum_test<float>(40960, 39, 1, 1); }
TEST(reduce_sum_layer, fp16_2x3x4_0) { reduce_sum_test<__half>(2, 3, 4, 0); }
TEST(reduce_sum_layer, fp16_2x3x4_1) { reduce_sum_test<__half>(2, 3, 4, 1); }
TEST(reduce_sum_layer, fp16_2x3x4_2) { reduce_sum_test<__half>(2, 3, 4, 2); }
TEST(reduce_sum_layer, fp16_40960x39x1_1) { reduce_sum_test<__half>(40960, 39, 1, 1); }
