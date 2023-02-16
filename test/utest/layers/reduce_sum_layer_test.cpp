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

#include <layers/reduce_sum_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

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
void reduce_sum_cpu(const T* input, T* output, core23::Shape shape, int64_t axis) {
  if (axis == 0) {
    if (shape.dims() == 1) {
      for (auto i = 0; i < shape.size(0); i++) {
        output[0] = input[i];
      }
    } else if (shape.dims() == 2) {
      for (auto k = 0; k < shape.size(1); k++) {
        output[k] = 0.0f;
        for (auto i = 0; i < shape.size(0); i++) {
          output[k] = output[k] + input[i * shape.size(1) + k];
        }
      }
    } else if (shape.dims() == 3) {
      for (auto j = 0; j < shape.size(1); j++) {
        for (auto k = 0; k < shape.size(2); k++) {
          output[j * shape.size(2) + k] = 0.0f;
          for (auto i = 0; i < shape.size(0); i++) {
            output[j * shape.size(2) + k] =
                output[j * shape.size(2) + k] +
                input[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (shape.dims() == 2) {
      for (auto i = 0; i < shape.size(0); i++) {
        output[i] = 0.0f;
        for (auto j = 0; j < shape.size(1); j++) {
          output[i] = output[i] + input[i * shape.size(1) + j];
        }
      }
    } else if (shape.dims() == 3) {
      for (auto i = 0; i < shape.size(0); i++) {
        for (auto k = 0; k < shape.size(2); k++) {
          output[i * shape.size(2) + k] = 0.0f;
          for (auto j = 0; j < shape.size(1); j++) {
            output[i * shape.size(2) + k] =
                output[i * shape.size(2) + k] +
                input[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (auto i = 0; i < shape.size(0); i++) {
      for (auto j = 0; j < shape.size(1); j++) {
        output[i * shape.size(1) + j] = 0.0f;
        for (auto k = 0; k < shape.size(2); k++) {
          output[i * shape.size(1) + j] =
              output[i * shape.size(1) + j] +
              input[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k];
        }
      }
    }
  }
}

template <typename T>
void reduce_sum_dgrad_cpu(const T* top_grad, T* dgrad, core23::Shape shape, int64_t axis) {
  if (axis == 0) {
    if (shape.dims() == 2) {
      for (auto j = 0; j < shape.size(1); j++) {
        for (auto i = 0; i < shape.size(0); i++) {
          dgrad[i * shape.size(1) + j] = top_grad[j];
        }
      }
    } else if (shape.dims() == 3) {
      for (auto j = 0; j < shape.size(1); j++) {
        for (auto k = 0; k < shape.size(2); k++) {
          for (auto i = 0; i < shape.size(0); i++) {
            dgrad[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k] =
                top_grad[j * shape.size(2) + k];
          }
        }
      }
    }
  } else if (axis == 1) {
    if (shape.dims() == 2) {
      for (auto i = 0; i < shape.size(0); i++) {
        for (auto j = 0; j < shape.size(1); j++) {
          dgrad[i * shape.size(1) + j] = top_grad[i];
        }
      }
    } else if (shape.dims() == 3) {
      for (auto i = 0; i < shape.size(0); i++) {
        for (auto k = 0; k < shape.size(2); k++) {
          for (auto j = 0; j < shape.size(1); j++) {
            dgrad[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k] =
                top_grad[i * shape.size(2) + k];
          }
        }
      }
    }
  } else if (axis == 2) {
    for (auto i = 0; i < shape.size(0); i++) {
      for (auto j = 0; j < shape.size(1); j++) {
        for (auto k = 0; k < shape.size(2); k++) {
          dgrad[i * shape.size(1) * shape.size(2) + j * shape.size(2) + k] =
              top_grad[i * shape.size(1) + j];
        }
      }
    }
  }
}

template <typename T>
void reduce_sum_test(int64_t batch_size, int64_t slot_num, int64_t embedding_vec_size,
                     int64_t axis) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  core23::Shape in_shape = {batch_size, slot_num, embedding_vec_size};  // 3D

  core23::Tensor bottom_tensor(tensor_params.shape(in_shape));
  core23::Tensor top_tensor;

  ReduceSumLayer<T> reduce_sum_layer(bottom_tensor, top_tensor, axis, test::get_default_gpu());

  reduce_sum_layer.initialize();

  auto in_size = in_shape.size();
  auto out_shape = top_tensor.shape();
  auto out_size = out_shape.size();

  std::vector<T> h_bottom(in_size);
  std::vector<T> h_top(out_size);
  std::vector<T> h_cpu_top(out_size);
  std::vector<T> h_gpu_dgrad(in_size);

  // fprop
  test::normal_sync_cpu(h_bottom.data(), h_bottom.size(), 0.f, 1.f, generator);

  core23::copy_sync(bottom_tensor.data(), h_bottom.data(), bottom_tensor.num_bytes(),
                    bottom_tensor.device(), core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  reduce_sum_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  reduce_sum_cpu<T>(h_bottom.data(), h_cpu_top.data(), in_shape, axis);

  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_cpu_top.data(), out_size, Eps<T>()));

  // bprop
  test::normal_sync_cpu(h_top.data(), h_top.size(), 0.f, 1.f, generator);
  core23::copy_sync(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                    core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  reduce_sum_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(h_gpu_dgrad.data(), bottom_tensor.data(), bottom_tensor.num_bytes(),
                    core23::DeviceType::CPU, bottom_tensor.device());

  reduce_sum_dgrad_cpu<T>(h_top.data(), h_bottom.data(), in_shape, axis);
  ASSERT_TRUE(test::compare_array_approx<T>(h_bottom.data(), h_gpu_dgrad.data(), in_size,
                                            Eps<T>()));  // compare dgrad
}

}  // namespace

TEST(reduce_sum_layer, fp32_2x3x4_0) { reduce_sum_test<float>(2, 3, 4, 0); }
TEST(reduce_sum_layer, fp32_2x3x4_1) { reduce_sum_test<float>(2, 3, 4, 1); }
TEST(reduce_sum_layer, fp32_2x3x4_2) { reduce_sum_test<float>(2, 3, 4, 2); }
TEST(reduce_sum_layer, fp32_23x100x18_1) { reduce_sum_test<float>(23, 100, 18, 1); }
TEST(reduce_sum_layer, fp32_40960x39x1_1) { reduce_sum_test<float>(40960, 39, 1, 1); }
TEST(reduce_sum_layer, fp16_2x3x4_0) { reduce_sum_test<__half>(2, 3, 4, 0); }
TEST(reduce_sum_layer, fp16_2x3x4_1) { reduce_sum_test<__half>(2, 3, 4, 1); }
TEST(reduce_sum_layer, fp16_2x3x4_2) { reduce_sum_test<__half>(2, 3, 4, 2); }
TEST(reduce_sum_layer, fp16_40960x39x1_1) { reduce_sum_test<__half>(40960, 39, 1, 1); }
