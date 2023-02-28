/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <base/debug/logger.hpp>
#include <core23/cuda_primitives.cuh>
#include <core23/macros.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_container.hpp>
#include <cstdint>
#include <random>
#include <utest/test_utils.hpp>

namespace {

using namespace HugeCTR::core23;

__global__ void transform_container_test_kernel(
    TensorContainer<int, 1, 2>::View output_tensor_container_view,
    TensorContainer<int, 1, 2>::View input_tensor_container_view) {
  int64_t h_base = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t w_base = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t i_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t h = h_base; h < output_tensor_container_view.size(0); h += gridDim.z) {
    for (int64_t w = w_base; w < output_tensor_container_view.size(1);
         w += blockDim.y * gridDim.y) {
      auto output_tensor_view = output_tensor_container_view[h][w];
      auto input_tensor_view = input_tensor_container_view[h][w];
      for (int64_t i = i_base; i < output_tensor_view.size(0); i += blockDim.x * gridDim.x) {
        output_tensor_view[i] = input_tensor_view[i];
      }
    }
  }
}

__global__ void transform_container_test_kernel(
    TensorContainer<int, 2, 1>::View output_tensor_container_view,
    TensorContainer<int, 2, 1>::View input_tensor_container_view) {
  int64_t h_base = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t w_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t t = blockIdx.z; t < output_tensor_container_view.size(0); t += gridDim.z) {
    auto output_tensor_view = output_tensor_container_view[t];
    auto input_tensor_view = input_tensor_container_view[t];
    for (int64_t h = h_base; h < output_tensor_view.size(0); h += blockDim.y * gridDim.y) {
      for (int64_t w = w_base; w < output_tensor_view.size(1); w += blockDim.x * gridDim.x) {
        output_tensor_view[h][w] = input_tensor_view[h][w];
      }
    }
  }
}

__global__ void transform_container_test_kernel(
    TensorContainer<int, 1, 1>::View output_tensor_container_view,
    TensorContainer<int, 1, 1>::View input_tensor_container_view) {
  int64_t i_base = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t t = blockIdx.z; t < output_tensor_container_view.size(0); t += gridDim.z) {
    auto output_tensor_view = output_tensor_container_view[t];
    auto input_tensor_view = input_tensor_container_view[t];
    for (int64_t i = i_base; i < output_tensor_view.size(0); i += blockDim.y * gridDim.y) {
      output_tensor_view[i] = input_tensor_view[i];
    }
  }
}

__global__ void transform_container_test_kernel(TensorView<int, 1> output_tensor_view,
                                                TensorView<int, 1> input_tensor_view) {
  int64_t i_base = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = i_base; i < output_tensor_view.size(0); i += blockDim.x * gridDim.x) {
    output_tensor_view[i] = input_tensor_view[i];
  }
}

}  // namespace

template <int64_t TensorDims, int64_t ContainerDims, bool Flatten = false>
void tensor_container_test_impl(Shape container_shape, Shape tensor_shape) {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, 1024);

  BufferParams buffer_params;

  std::vector<Tensor> input_tensors, output_tensors;
  TensorParams tensor_params = TensorParams().shape(tensor_shape).data_type(ScalarType::Int32);
  for (int64_t t = 0; t < container_shape.size(); t++) {
    input_tensors.emplace_back(tensor_params);
  }
  for (int64_t t = 0; t < container_shape.size(); t++) {
    output_tensors.emplace_back(tensor_params);
  }

  TensorContainer<int, TensorDims, ContainerDims> input_tensor_container(std::move(input_tensors),
                                                                         Shape(container_shape));
  TensorContainer<int, TensorDims, ContainerDims> input_tensor_container_copy0 =
      input_tensor_container;

  for (int64_t t = 0; t < container_shape.size(); t++) {
    std::vector<int> h_ins(tensor_shape.size());
    for (size_t i = 0; i < h_ins.size(); i++) {
      h_ins[i] = uniform_dist(e);
    }
    if constexpr (ContainerDims == 1) {
      copy_sync(input_tensor_container[t].data(), h_ins.data(), h_ins.size() * sizeof(int),
                input_tensor_container[t].device(), DeviceType::CPU);
    } else {
      auto w = t % container_shape.size(1);
      auto h = t / container_shape.size(1);
      copy_sync(input_tensor_container[h][w].data(), h_ins.data(), h_ins.size() * sizeof(int),
                input_tensor_container[h][w].device(), DeviceType::CPU);
    }
  }

  TensorContainer<int, TensorDims, ContainerDims> input_tensor_container_copy1 =
      input_tensor_container;

  TensorContainer<int, TensorDims, ContainerDims> output_tensor_container_original(
      output_tensors, Shape(container_shape));
  TensorContainer<int, TensorDims, ContainerDims> output_tensor_container(
      output_tensor_container_original);

  for (int64_t t = 0; t < container_shape.size(); t++) {
    if constexpr (ContainerDims == 1) {
      EXPECT_TRUE(input_tensor_container[t].data() == input_tensor_container_copy0[t].data());
      EXPECT_TRUE(input_tensor_container[t].data() == input_tensor_container_copy1[t].data());
      EXPECT_TRUE(input_tensor_container[t].device() == input_tensor_container_copy0[t].device());
      EXPECT_TRUE(input_tensor_container[t].device() == input_tensor_container_copy1[t].device());
    } else {
      auto w = t % container_shape.size(1);
      auto h = t / container_shape.size(1);
      EXPECT_TRUE(input_tensor_container[h][w].data() == input_tensor_container_copy0[h][w].data());
      EXPECT_TRUE(input_tensor_container[h][w].data() == input_tensor_container_copy1[h][w].data());
      EXPECT_TRUE(input_tensor_container[h][w].device() ==
                  input_tensor_container_copy0[h][w].device());
      EXPECT_TRUE(input_tensor_container[h][w].device() ==
                  input_tensor_container_copy1[h][w].device());
    }
  }

  CUDAStream stream(cudaStreamDefault, 0);
  if constexpr (Flatten && TensorDims == 1 && ContainerDims == 1) {
    dim3 block(512, 1, 1);
    dim3 grid(64);
    transform_container_test_kernel<<<grid, block, 0, stream()>>>(output_tensor_container.flatten(),
                                                                  input_tensor_container.flatten());
  } else {
    dim3 block(32, 16, 1);
    dim3 grid(8, 4, 2);
    transform_container_test_kernel<<<grid, block, 0, stream()>>>(output_tensor_container.view(),
                                                                  input_tensor_container.view());
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  if constexpr (Flatten && TensorDims == 1 && ContainerDims == 1) {
    TensorContainer<int, TensorDims, ContainerDims> input_tensor_container_copy2 =
        input_tensor_container;
    for (int64_t t = 0; t < container_shape.size(); t++) {
      EXPECT_TRUE(input_tensor_container[t].data() == input_tensor_container_copy2[t].data());
      EXPECT_TRUE(input_tensor_container[t].device() == input_tensor_container_copy2[t].device());
    }
  }

  for (int64_t t = 0; t < container_shape.size(); t++) {
    std::vector<int> h_ins(tensor_shape.size());
    std::vector<int> h_outs(tensor_shape.size());
    if constexpr (ContainerDims == 1) {
      copy_sync(h_ins.data(), input_tensor_container[t].data(), h_ins.size() * sizeof(int),
                DeviceType::CPU, input_tensor_container[t].device());
      copy_sync(h_outs.data(), output_tensor_container[t].data(), h_outs.size() * sizeof(int),
                DeviceType::CPU, input_tensor_container[t].device());
    } else {
      auto w = t % container_shape.size(1);
      auto h = t / container_shape.size(1);
      copy_sync(h_ins.data(), input_tensor_container[h][w].data(), h_ins.size() * sizeof(int),
                DeviceType::CPU, input_tensor_container[h][w].device());
      copy_sync(h_outs.data(), output_tensor_container[h][w].data(), h_outs.size() * sizeof(int),
                DeviceType::CPU, input_tensor_container[h][w].device());
    }
    ASSERT_TRUE(
        HugeCTR::test::compare_array_approx<int>(h_outs.data(), h_ins.data(), h_outs.size(), 0));
  }
}

TEST(test_core23, tensor_container_test_1d_2d) {
  tensor_container_test_impl<1, 2>({4, 2}, {1024 * 512});
}
TEST(test_core23, tensor_container_test_2d_1d) {
  tensor_container_test_impl<2, 1>({4 * 2}, {1024, 512});
}
TEST(test_core23, tensor_container_test_1d_1d) {
  tensor_container_test_impl<1, 1>({4 * 2}, {1024 * 512});
}

TEST(test_core23, tensor_container_test_flatten_1d_success) {
  tensor_container_test_impl<1, 1, true>({4 * 2}, {1024 * 512});
}

TEST(test_core23, tensor_container_test_flatten_1d) {
  tensor_container_test_impl<1, 1, true>({4 * 2}, {1024 * 512});
}
