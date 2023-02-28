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

#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <layers/add_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

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

template <typename Vector, typename T>
void add_cpu(Vector input, T *output, size_t size, size_t num) {
  for (auto i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += input[j][i];
    }
    output[i] = tmp;
  }
}

template <>
void add_cpu(std::vector<std::vector<__half>> input, __half *output, size_t size, size_t num) {
  for (auto i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += __half2float(input[j][i]);
    }
    output[i] = __float2half(tmp);
  }
}

template <typename Vector, typename T>
void add_dgrad_cpu(const T *top_grad, Vector dgrad, size_t size, size_t num) {
  for (auto i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      dgrad[j][i] = top_grad[i];
    }
  }
}

template <typename T>
void add_test(int64_t batch_size, int64_t slot_num, int64_t embedding_vec_size, int64_t num) {
  constexpr bool use_mixed_precision = std::is_same_v<T, __half>;

  core23::Shape shape_bottom = {batch_size, slot_num, embedding_vec_size};
  core23::Shape shape_top = {batch_size, slot_num, embedding_vec_size};
  auto size = batch_size * slot_num * embedding_vec_size;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(device)
          .data_type(use_mixed_precision ? core23::ScalarType::Half : core23::ScalarType::Float)
          .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> bottom_tensors;
  for (auto i = 0; i < num; i++) {
    bottom_tensors.emplace_back(tensor_params.shape(shape_bottom));
  }
  core23::Tensor top_tensor(tensor_params.shape(shape_top));

  AddLayer<T> add_layer(bottom_tensors, top_tensor, test::get_default_gpu());

  add_layer.initialize();

  std::vector<std::vector<T>> h_bottoms(num);
  for (auto i = 0; i < num; i++) {
    h_bottoms[i] = std::vector<T>(size);
  }
  std::vector<T> h_top(size);
  std::vector<T> h_cpu_top(size);
  std::vector<std::vector<T>> h_gpu_dgrads(num);
  for (auto i = 0; i < num; i++) {
    h_gpu_dgrads[i] = std::vector<T>(size);
  }

  // fprop
  for (auto i = 0; i < num; i++) {
    test::normal_sync_cpu(h_bottoms[i].data(), h_bottoms[i].size(), 0.f, 1.f, generator);
    core23::copy_sync(bottom_tensors[i].data(), h_bottoms[i].data(), bottom_tensors[i].num_bytes(),
                      bottom_tensors[i].device(), core23::DeviceType::CPU);
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  add_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  add_cpu(h_bottoms.data(), h_cpu_top.data(), size, num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_cpu_top.data(), size, Eps<T>::value()));

  // bprop
  for (auto i = 0; i < num; i++) {
    test::normal_sync_cpu(h_bottoms[i].data(), h_bottoms[i].size(), 0.f, 1.f, generator);
    core23::copy_sync(bottom_tensors[i].data(), h_bottoms[i].data(), bottom_tensors[i].num_bytes(),
                      bottom_tensors[i].device(), core23::DeviceType::CPU);
  }
  test::normal_sync_cpu(h_top.data(), h_top.size(), 0.f, 1.f, generator);
  core23::copy_sync(top_tensor.data(), h_top.data(), top_tensor.num_bytes(), top_tensor.device(),
                    core23::DeviceType::CPU);

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  add_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (auto i = 0; i < num; i++) {
    core23::copy_sync(h_gpu_dgrads[i].data(), bottom_tensors[i].data(),
                      bottom_tensors[i].num_bytes(), core23::DeviceType::CPU,
                      bottom_tensors[i].device());
  }

  add_dgrad_cpu(h_top.data(), h_bottoms.data(), size, num);
  for (auto i = 0; i < num; i++) {
    ASSERT_TRUE(test::compare_array_approx<T>(h_bottoms[i].data(), h_gpu_dgrads[i].data(), size,
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
