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
#include <cmath>
#include <core23/cuda_stream.hpp>
#include <core23/curand_generator.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/low_level_primitives.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_params.hpp>
#include <random>
#include <utest/test_utils.hpp>
#include <vector>

namespace {

using namespace HugeCTR::core23;

void fill_test_impl(Device device, bool async) {
  TensorParams params = TensorParams().shape({128, 256}).data_type(ScalarType::Int32);
  Tensor tensor(params.device(device));

  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist(0, tensor.num_elements());
  int val = uniform_dist(e);

  if (async) {
    CUDAStream stream(cudaStreamDefault, 0);
    fill_async<int>(tensor.data<int>(), tensor.num_elements(), val, device, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  } else {
    fill_sync<int>(tensor.data<int>(), tensor.num_elements(), val, device);
  }

  std::vector<int> h_out(tensor.num_elements());
  copy_sync(h_out.data(), tensor.data(), tensor.num_bytes(), DeviceType::CPU, device);
  std::vector<int> h_ref(tensor.num_elements(), val);

  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_out.data(), h_ref.data(), h_out.size(), 0));
}

void copy_test_impl(Device dst_device, Device src_device, bool async) {
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
    copy_async(dst_tensor.data(), src_tensor.data(), src_tensor.num_bytes(), dst_device, src_device,
               stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  } else {
    copy_sync(dst_tensor.data(), src_tensor.data(), src_tensor.num_bytes(), dst_device, src_device);
  }

  std::vector<int> h_dst(dst_tensor.num_elements());
  copy_sync(h_dst.data(), dst_tensor.data(), dst_tensor.num_bytes(), DeviceType::CPU, dst_device);
  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<int>(h_dst.data(), h_src.data(), h_dst.size(), 0));
}

template <typename DstType, typename SrcType>
void convert_test_impl(Device dst_device, Device src_device) {
  TensorParams params = TensorParams().shape({512, 512}).data_type(ToScalarType<SrcType>::value);
  Tensor dst_tensor(params.device(dst_device).data_type(ToScalarType<DstType>::value));
  Tensor src_tensor(params.device(src_device));

  std::vector<SrcType> h_src(src_tensor.num_elements());
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_real_distribution<SrcType> uniform_dist(-0.1, 0.1);
  for (size_t i = 0; i < h_src.size(); i++) {
    h_src[i] = uniform_dist(e);
  }
  copy_sync(src_tensor.data(), h_src.data(), src_tensor.num_bytes(), src_device, DeviceType::CPU);

  CUDAStream stream(cudaStreamDefault, 0);
  convert_async<DstType, SrcType>(dst_tensor.data<DstType>(), src_tensor.data<SrcType>(),
                                  src_tensor.num_elements(), dst_device, src_device, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));

  std::vector<DstType> h_dst(dst_tensor.num_elements());
  copy_sync(h_dst.data(), dst_tensor.data(), dst_tensor.num_bytes(), DeviceType::CPU, dst_device);
  std::vector<DstType> h_ref(dst_tensor.num_elements());
  for (size_t i = 0; i < h_ref.size(); i++) {
    h_ref[i] = TypeConverter<DstType, SrcType>::value(h_src[i]);
  }

  ASSERT_TRUE(
      HugeCTR::test::compare_array_approx<DstType>(h_dst.data(), h_ref.data(), h_dst.size(), 1e-8));
}

enum class TestRandomType {
  Uniform,
  Normal,
  XavierUniform,
  XavierNormal,
};

template <typename Type>
void random_test_common_impl(Device device, TestRandomType type) {
  TensorParams params = TensorParams().shape({1024, 1024}).data_type(ToScalarType<Type>::value);
  Tensor tensor(params.device(device));

  Type a = TypeConverter<Type, float>::value(1.f);
  Type b = TypeConverter<Type, float>::value(2.f);
  Type fan_in = TypeConverter<Type, float>::value(params.shape().size());
  Type fan_out = TypeConverter<Type, float>::value(params.shape().size());

  CUDAStream stream(cudaStreamDefault, 0);
  CURANDGenerator generator(device);

  Type mean_in = 0.f;
  Type stddev_in = 0.f;
  if (type == TestRandomType::Uniform) {
    mean_in = (b + a) / TypeConverter<Type, float>::value(2.f);
    stddev_in = std::sqrt((b - a) * (b - a) / TypeConverter<Type, float>::value(12.f));
    uniform_async<Type>(tensor.data<Type>(), tensor.num_elements(), a, b, device, generator,
                        stream);
    cudaStreamSynchronize(stream());
  } else if (type == TestRandomType::Normal) {
    mean_in = TypeConverter<Type, float>::value(0.f);
    stddev_in = TypeConverter<Type, float>::value(0.05f);
    normal_async<Type>(tensor.data<Type>(), tensor.num_elements(), mean_in, stddev_in, device,
                       generator, stream);
  } else if (type == TestRandomType::XavierUniform) {
    mean_in = TypeConverter<Type, float>::value(0.f);
    stddev_in = TypeConverter<Type, float>::value(0.f);
    xavier_uniform_async<Type>(tensor.data<Type>(), tensor.num_elements(), fan_in, fan_out, device,
                               generator, stream);
  } else if (type == TestRandomType::XavierNormal) {
    mean_in = TypeConverter<Type, float>::value(0.f);
    stddev_in = TypeConverter<Type, float>::value(std::sqrt(2.f / (fan_in + fan_out)));
    xavier_normal_async<Type>(tensor.data<Type>(), tensor.num_elements(), fan_in, fan_out, device,
                              generator, stream);
  }
  cudaStreamSynchronize(stream());

  std::vector<Type> h_out(tensor.num_elements());
  copy_sync(h_out.data(), tensor.data(), tensor.num_bytes(), DeviceType::CPU, device);

  int64_t n = h_out.size();
  Type mean_out =
      std::accumulate(h_out.begin(), h_out.end(), TypeConverter<Type, float>::value(0.f)) / n;
  Type stddev_out =
      std::sqrt(std::accumulate(h_out.begin(), h_out.end(), TypeConverter<Type, float>::value(0.f),
                                [mean_out, n](Type accum, const Type& val) {
                                  return accum + ((val - mean_out) * (val - mean_out)) / (n - 1);
                                }));
  EXPECT_TRUE(std::abs(mean_out - mean_in) < 1e-3);
  EXPECT_TRUE(std::abs(stddev_out - stddev_in) < 1e-3);
}

}  // namespace

TEST(test_core23, gpu_fill_sync_test) {
  Device device(DeviceType::GPU, 0);
  fill_test_impl(device, false);
}

TEST(test_core23, cpu_fill_sync_test) {
  Device device(DeviceType::CPU);
  fill_test_impl(device, false);
}

TEST(test_core23, gpu_fill_async_test) {
  Device device(DeviceType::GPU, 0);
  fill_test_impl(device, true);
}

TEST(test_core23, cpu_fill_async_test) {
  Device device(DeviceType::CPU);
  fill_test_impl(device, true);
}

TEST(test_core23, intra_gpu_copy_sync_test) {
  Device device(DeviceType::GPU, 0);
  copy_test_impl(device, device, false);
}

TEST(test_core23, intra_cpu_copy_sync_test) {
  Device device(DeviceType::CPU);
  copy_test_impl(device, device, false);
}

TEST(test_core23, cpu_to_gpu_copy_sync_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  copy_test_impl(dst_device, src_device, false);
}

TEST(test_core23, gpu_to_cpu_copy_sync_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  copy_test_impl(dst_device, src_device, false);
}

TEST(test_core23, intra_gpu_copy_async_test) {
  Device device(DeviceType::GPU, 0);
  copy_test_impl(device, device, true);
}

TEST(test_core23, intra_cpu_copy_async_test) {
  Device device(DeviceType::CPU, 0);
  copy_test_impl(device, device, true);
}

TEST(test_core23, cpu_to_gpu_copy_async_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  copy_test_impl(dst_device, src_device, true);
}

TEST(test_core23, gpu_to_cpu_copy_async_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  copy_test_impl(dst_device, src_device, true);
}

TEST(test_core23, intra_gpu_float_to_half_convert_async_test) {
  Device device(DeviceType::GPU, 0);
  convert_test_impl<__half, float>(device, device);
}

TEST(test_core23, intra_cpu_float_to_half_convert_async_test) {
  Device device(DeviceType::CPU);
  convert_test_impl<__half, float>(device, device);
}

TEST(test_core23, cpu_to_gpu_float_to_half_convert_async_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  convert_test_impl<__half, float>(dst_device, src_device);
}

TEST(test_core23, gpu_to_cpu_float_to_half_convert_async_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  convert_test_impl<__half, float>(dst_device, src_device);
}

TEST(test_core23, intra_gpu_float_to_int_convert_async_test) {
  Device device(DeviceType::GPU, 0);
  convert_test_impl<int32_t, float>(device, device);
}

TEST(test_core23, intra_cpu_float_to_int_convert_async_test) {
  Device device(DeviceType::CPU);
  convert_test_impl<int32_t, float>(device, device);
}

TEST(test_core23, cpu_to_gpu_float_to_int_convert_async_test) {
  Device dst_device(DeviceType::GPU, 0);
  Device src_device(DeviceType::CPU);
  convert_test_impl<int32_t, float>(dst_device, src_device);
}

TEST(test_core23, gpu_to_cpu_float_to_int_convert_async_test) {
  Device dst_device(DeviceType::CPU);
  Device src_device(DeviceType::GPU, 0);
  convert_test_impl<int32_t, float>(dst_device, src_device);
}

TEST(test_core23, gpu_float_uniform_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<float>(device, TestRandomType::Uniform);
}

TEST(test_core23, gpu_double_uniform_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<double>(device, TestRandomType::Uniform);
}

TEST(test_core23, cpu_float_uniform_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<float>(device, TestRandomType::Uniform);
}

TEST(test_core23, cpu_double_uniform_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<double>(device, TestRandomType::Uniform);
}

TEST(test_core23, gpu_float_normal_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<float>(device, TestRandomType::Normal);
}

TEST(test_core23, gpu_double_normal_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<double>(device, TestRandomType::Normal);
}

TEST(test_core23, cpu_float_normal_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<float>(device, TestRandomType::Normal);
}

TEST(test_core23, cpu_double_normal_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<double>(device, TestRandomType::Normal);
}

TEST(test_core23, gpu_float_xavier_uniform_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<float>(device, TestRandomType::XavierUniform);
}

TEST(test_core23, gpu_double_xavier_uniform_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<double>(device, TestRandomType::XavierUniform);
}

TEST(test_core23, cpu_float_xavier_uniform_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<float>(device, TestRandomType::XavierUniform);
}

TEST(test_core23, cpu_double_xavier_uniform_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<double>(device, TestRandomType::XavierUniform);
}

TEST(test_core23, gpu_float_xavier_normal_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<float>(device, TestRandomType::XavierNormal);
}

TEST(test_core23, gpu_double_xavier_normal_async_test) {
  Device device(DeviceType::GPU, 0);
  random_test_common_impl<double>(device, TestRandomType::XavierNormal);
}

TEST(test_core23, cpu_float_xavier_normal_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<float>(device, TestRandomType::XavierNormal);
}

TEST(test_core23, cpu_double_xavier_normal_async_test) {
  Device device(DeviceType::CPU);
  random_test_common_impl<double>(device, TestRandomType::XavierNormal);
}
