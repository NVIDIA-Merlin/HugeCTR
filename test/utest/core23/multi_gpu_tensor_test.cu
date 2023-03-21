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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common.hpp>
#include <core23/cuda_primitives.cuh>
#include <core23/device_guard.hpp>
#include <core23/logger.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <cstdint>
#include <random>
#include <thread>
#include <utest/test_utils.hpp>
#include <vector>

namespace {

using namespace HugeCTR::core23;

void multi_gpu_tensor_test_impl() {
  int64_t device_count = Device::count();

  Shape shape({1024, 128});

  std::vector<int> h_input(shape.size());
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, shape.size());
  for (size_t i = 0; i < h_input.size(); i++) {
    h_input[i] = uniform_dist(e);
  }

  std::vector<Tensor> d_outputs;
  std::vector<std::thread> threads;
  for (int64_t d = 0; d < device_count; d++) {
    Device device(DeviceType::GPU, d);
    DeviceGuard device_guard(device);
    CUDAStream stream(cudaStreamDefault, 0);
    TensorParams tensor_params =
        TensorParams().device(device).data_type(ScalarType::Int32).shape(shape).stream(stream);
    Tensor input_tensor(tensor_params);
    Tensor output_tensor(tensor_params);
    d_outputs.push_back(output_tensor);

    copy_sync(input_tensor.data(), h_input.data(), input_tensor.num_bytes(), input_tensor.device(),
              DeviceType::CPU);

    threads.emplace_back([device, input_tensor, output_tensor]() {
      auto stream = output_tensor.my_params().stream();
      DeviceGuard device_guard(device);
      dim3 block(32, 32, 1);
      dim3 grid((output_tensor.size(1) + block.x - 1) / block.x,
                (output_tensor.size(0) + block.y - 1) / block.y);
      copy_kernel<int>
          <<<grid, block, 0, stream()>>>(input_tensor.view<int, 2>(), output_tensor.view<int, 2>());
      HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (auto& tensor : d_outputs) {
    std::vector<int> h_output(tensor.num_elements());
    copy_sync(h_output.data(), tensor.data(), tensor.num_bytes(), DeviceType::CPU, tensor.device());
    ASSERT_TRUE(HugeCTR::test::compare_array_approx<int>(h_output.data(), h_input.data(),
                                                         h_output.size(), 0));
  }
}

}  // namespace

TEST(test_core23, multi_gpu_tensor_test) { multi_gpu_tensor_test_impl(); }
