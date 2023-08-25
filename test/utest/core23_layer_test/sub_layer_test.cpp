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

#include <layers/sub_layer.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>
#include <vector>

using namespace HugeCTR;

namespace {

const float eps = 1e-6f;

template <typename Vector, typename T>
void sub_cpu(Vector input, T *output, size_t size) {
  for (auto i = 0; i < size; i++) output[i] = input[0][i] - input[1][i];
}

template <typename Vector, typename T>
void sub_dgrad_cpu(const T *top_grad, Vector dgrad, size_t size) {
  for (auto i = 0; i < size; i++) {
    dgrad[0][i] = top_grad[i];
    dgrad[1][i] = 0.0 - top_grad[i];
  }
}

template <typename T>
void sub_test(int64_t batch_size, int64_t slot_num, int64_t embedding_vec_size, int64_t num) {
  core23::Shape shape_bottom = {batch_size, slot_num, embedding_vec_size};
  core23::Shape shape_top = {batch_size, slot_num, embedding_vec_size};
  auto size = batch_size * slot_num * embedding_vec_size;

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params = core23::TensorParams()
                                           .device(device)
                                           .data_type(core23::ScalarType::Float)
                                           .buffer_channel(core23::GetRandomBufferChannel());

  std::vector<core23::Tensor> bottom_tensors;
  for (auto i = 0; i < num; i++) {
    bottom_tensors.emplace_back(tensor_params.shape(shape_bottom));
  }
  core23::Tensor top_tensor(tensor_params.shape(shape_top));

  SubLayer<T> sub_layer(bottom_tensors, top_tensor, test::get_default_gpu());

  sub_layer.initialize();

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
  sub_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  core23::copy_sync(h_top.data(), top_tensor.data(), top_tensor.num_bytes(),
                    core23::DeviceType::CPU, top_tensor.device());

  sub_cpu(h_bottoms.data(), h_cpu_top.data(), size);
  ASSERT_TRUE(test::compare_array_approx<T>(h_top.data(), h_cpu_top.data(), size, eps));

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
  sub_layer.bprop();  // compute wgrad and dgrad
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  for (auto i = 0; i < num; i++) {
    core23::copy_sync(h_gpu_dgrads[i].data(), bottom_tensors[i].data(),
                      bottom_tensors[i].num_bytes(), core23::DeviceType::CPU,
                      bottom_tensors[i].device());
  }

  sub_dgrad_cpu(h_top.data(), h_bottoms.data(), size);
  for (auto i = 0; i < num; i++) {
    ASSERT_TRUE(test::compare_array_approx<T>(h_bottoms[i].data(), h_gpu_dgrads[i].data(), size,
                                              eps));  // compare dgrad
  }
}

}  // namespace

TEST(sub_layer, fp32) { sub_test<float>(40960, 1, 1, 2); }
TEST(sub_layer, fp16) { sub_test<float>(40960, 2, 110, 2); }
