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

#include <base/debug/logger.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <core23/tensor_params.hpp>
#include <random>
#include <utest/test_utils.hpp>
#include <vector>

namespace {

using namespace HugeCTR::core23;

void tensor_zeros_test_impl(Device device, bool async) {
  TensorParams params = TensorParams().shape({128, 256}).data_type(ScalarType::Int32);
  Tensor tensor(params.device(device));

  if (async) {
    CUDAStream stream(cudaStreamDefault, 0);
    zeros_async(tensor, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  } else {
    zeros_sync(tensor);
  }

  std::vector<int> h_out(tensor.num_elements());
  copy_sync(h_out.data(), tensor.data(), tensor.num_bytes(), DeviceType::CPU, device);
  std::vector<int> h_ref(tensor.num_elements(), 0);
  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_out.data(), h_ref.data(), h_out.size(), 0));
}

void tensor_copy_test_impl(Device dst_device, Device src_device, bool async) {
  TensorParams params = TensorParams().shape({128, 256}).data_type(ScalarType::Int32);
  Tensor dst_tensor(params.device(dst_device));
  Tensor src_tensor(params.device(src_device));

  std::vector<int> h_src(src_tensor.num_elements());
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, h_src.size());
  for (size_t i = 0; i < h_src.size(); i++) {
    h_src[i] = uniform_dist(e);
  }
  copy_sync(src_tensor.data(), h_src.data(), src_tensor.num_bytes(), src_device, DeviceType::CPU);

  if (async) {
    CUDAStream stream(cudaStreamDefault, 0);
    copy_async(dst_tensor, src_tensor, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  } else {
    copy_sync(dst_tensor, src_tensor);
  }

  std::vector<int> h_dst(dst_tensor.num_elements());
  copy_sync(h_dst.data(), dst_tensor.data(), dst_tensor.num_bytes(), DeviceType::CPU, dst_device);
  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_dst.data(), h_src.data(), h_dst.size(), 0));
}

void tensor_convert_test_impl(DataType dst_type, DataType src_type, Device dst_device,
                              Device src_device) {
  TensorParams params = TensorParams().shape({128, 256}).data_type(src_type).device(src_device);
  Tensor dst_tensor(params.device(dst_device).data_type(dst_type));
  Tensor src_tensor(params);

  std::vector<float> h_src(src_tensor.num_elements());
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_real_distribution<float> uniform_dist(-0.1, 0.1);
  for (size_t i = 0; i < h_src.size(); i++) {
    h_src[i] = uniform_dist(e);
  }
  copy_sync(src_tensor.data(), h_src.data(), src_tensor.num_bytes(), src_device, DeviceType::CPU);

  CUDAStream stream(cudaStreamDefault, 0);
  convert_async(dst_tensor, src_tensor, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  std::vector<int> h_dst(dst_tensor.num_elements());
  copy_sync(h_dst.data(), dst_tensor.data(), dst_tensor.num_bytes(), DeviceType::CPU, dst_device);
  std::vector<int> h_ref(dst_tensor.num_elements());
  for (size_t i = 0; i < h_ref.size(); i++) {
    h_ref[i] = TypeConverter<int32_t, float>::value(h_src[i]);
  }
  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int32_t>(h_dst.data(), h_ref.data(), h_dst.size(), 0));
}

}  // namespace

TEST(test_core23, gpu_tensor_zeros_sync_test) {
  Device device(DeviceType::GPU, 0);
  tensor_zeros_test_impl(device, false);
}

TEST(test_core23, cpu_tensor_zeros_sync_test) {
  Device device(DeviceType::CPU);
  tensor_zeros_test_impl(device, false);
}

TEST(test_core23, gpu_tensor_zeros_async_test) {
  Device device(DeviceType::GPU, 0);
  tensor_zeros_test_impl(device, true);
}

TEST(test_core23, cpu_tensor_zeros_async_test) {
  Device device(DeviceType::CPU);
  tensor_zeros_test_impl(device, true);
}

TEST(test_core23, intra_gpu_tensor_copy_sync_test) {
  Device device(DeviceType::GPU, 0);
  tensor_copy_test_impl(device, device, false);
}

TEST(test_core23, intra_cpu_tensor_copy_sync_test) {
  Device device(DeviceType::CPU);
  tensor_copy_test_impl(device, device, false);
}

TEST(test_core23, cpu_to_gpu_tensor_copy_sync_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  tensor_copy_test_impl(dst_device, src_device, false);
}

TEST(test_core23, gpu_to_cpu_tensor_copy_sync_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  tensor_copy_test_impl(dst_device, src_device, false);
}

TEST(test_core23, intra_gpu_tensor_copy_async_test) {
  Device device(DeviceType::GPU, 0);
  tensor_copy_test_impl(device, device, true);
}

TEST(test_core23, intra_cpu_tensor_copy_async_test) {
  Device device(DeviceType::CPU, 0);
  tensor_copy_test_impl(device, device, true);
}

TEST(test_core23, cpu_to_gpu_tensor_copy_async_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  tensor_copy_test_impl(dst_device, src_device, true);
}

TEST(test_core23, gpu_to_cpu_tensor_copy_async_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  tensor_copy_test_impl(dst_device, src_device, true);
}

TEST(test_core23, inter_gpu_tensor_convert_async_test) {
  DataType dst_type(ScalarType::Int32);
  DataType src_type(ScalarType::Float);
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::GPU, 0);
  tensor_convert_test_impl(dst_type, src_type, dst_device, src_device);
}

TEST(test_core23, inter_cpu_tensor_convert_async_test) {
  DataType dst_type(ScalarType::Int32);
  DataType src_type(ScalarType::Float);
  Device dst_device(DeviceType::CPU, 0);
  Device src_device(DeviceType::CPU, 0);
  tensor_convert_test_impl(dst_type, src_type, dst_device, src_device);
}

TEST(test_core23, gpu_to_cpu_tensor_convert_async_test) {
  DataType dst_type(ScalarType::Int32);
  DataType src_type(ScalarType::Float);
  Device dst_device(DeviceType::CPU, 0);
  Device src_device(DeviceType::GPU, 0);
  tensor_convert_test_impl(dst_type, src_type, dst_device, src_device);
}

TEST(test_core23, cpu_to_gpu_tensor_convert_async_test) {
  DataType dst_type(ScalarType::Int32);
  DataType src_type(ScalarType::Float);
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU, 0);
  tensor_convert_test_impl(dst_type, src_type, dst_device, src_device);
}
