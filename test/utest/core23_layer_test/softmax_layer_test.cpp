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

#include <layers/softmax_layer.hpp>
#include <utest/test_utils.hpp>
#include <vector>

using namespace HugeCTR;
using namespace std;

namespace {

template <typename T>
T get_eps(bool use_tf32 = false);

template <>
float get_eps(bool use_tf32) {
  return (use_tf32 ? 5e-1 : 1e-3);
}

template <>
__half get_eps(bool use_tf32) {
  return __float2half(1);
}
template <typename T>
void sum_ex_cpu(T* top, int embedding_vector_size, int dim0, T* workspace) {
  // sum(e^xi) i = [0, embedding_vector_size -1];
  for (int i = 0; i < dim0; i++) {
    workspace[i] = TypeConvert<T, float>::convert(0.f);
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      workspace[i] =
          TypeConvert<T, float>::convert(TypeConvert<float, T>::convert(top[offset + j]) +
                                         TypeConvert<float, T>::convert(workspace[i]));
    }
  }
}

template <typename T>
void ex_cpu(T* top, const T* bottom, int len) {
  // e^xi
  for (int i = 0; i < len; i++) {
    top[i] = TypeConvert<T, float>::convert(expf(TypeConvert<float, T>::convert(bottom[i])));
  }
}

template <typename T>
void sum_grad_softmax(const T* d_top, const T* softmax_out, int embedding_vector_size, int dim0,
                      T* workspace) {
  for (int i = 0; i < dim0; i++) {
    float grad_sum = 0.0;
    int offset = i * embedding_vector_size;
    for (int j = 0; j < embedding_vector_size; j++) {
      grad_sum += (TypeConvert<float, T>::convert(d_top[offset + j]) *
                   TypeConvert<float, T>::convert(softmax_out[offset + j]));
    }
    workspace[i] = TypeConvert<T, float>::convert(grad_sum);
    // printf("CPU grad_sum %d: %f\n", i, workspace[i]);
  }
}

template <typename T>
void softmax_fprop_cpu(T* top, const T* bottom, int len, int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T* workspace = new T[dim0];
  // e^xi
  ex_cpu(top, bottom, len);
  // sum(e^xi) i = [0, embedding_vector_size -1];
  sum_ex_cpu(top, embedding_vector_size, dim0, workspace);
  // softmax : e^xi / sum(e^xi); i = [0, len - 1];
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      top[index] = TypeConvert<T, float>::convert(TypeConvert<float, T>::convert(top[index]) /
                                                  TypeConvert<float, T>::convert(workspace[i]));
    }
  }
  delete[] workspace;
}

template <typename T>
void softmax_bprop_cpu(T* d_bottom, const T* d_top, const T* softmax_out, int len,
                       int embedding_vector_size) {
  int dim0 = len / embedding_vector_size;
  T* workspace = new T[dim0];

  sum_grad_softmax(d_top, softmax_out, embedding_vector_size, dim0, workspace);
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < embedding_vector_size; j++) {
      int index = i * embedding_vector_size + j;
      d_bottom[index] =
          TypeConvert<T, float>::convert(TypeConvert<float, T>::convert(softmax_out[index]) *
                                         (TypeConvert<float, T>::convert(d_top[index]) -
                                          TypeConvert<float, T>::convert(workspace[i])));
      // d_bottom[index] = workspace[i];
    }
  }
  delete[] workspace;
}

template <typename T>
void softmax_test(int64_t dim0, int64_t embedding_vector_size) {
  std::vector<int64_t> dims = {dim0, embedding_vector_size};

  auto device = core23::Device::current();
  core23::CURANDGenerator generator(core23::DeviceType::CPU);

  core23::TensorParams tensor_params = core23::TensorParams()
                                           .device(device)
                                           .data_type(core23::ToScalarType<T>::value)
                                           .buffer_channel(core23::GetRandomBufferChannel());

  core23::Tensor bottom_tensor(tensor_params.shape(dims));
  core23::Tensor top_tensor(tensor_params.shape(dims));

  SoftmaxLayer<T> softmax_layer(bottom_tensor, top_tensor, test::get_default_gpu());

  softmax_layer.initialize();

  const auto len = dim0 * embedding_vector_size;

  std::unique_ptr<T[]> h_bottom(new T[len]);
  std::unique_ptr<T[]> h_top(new T[len]);
  std::unique_ptr<T[]> h_softmax_out(new T[len]);
  std::unique_ptr<T[]> d2h_top(new T[len]);
  std::unique_ptr<T[]> h_bottom_grad(new T[len]);
  std::unique_ptr<T[]> d2h_bottom_grad(new T[len]);

  test::normal_sync_cpu(h_bottom.get(), len, 0.f, 1.f, generator);
  // fprop
  core23::copy_sync(bottom_tensor.data(), h_bottom.get(), len * sizeof(T), bottom_tensor.device(),
                    core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  softmax_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_top.get(), top_tensor.data(), len * sizeof(T), core23::DeviceType::CPU,
                    top_tensor.device());
  softmax_fprop_cpu<T>(h_top.get(), h_bottom.get(), len, embedding_vector_size);
  ASSERT_TRUE(test::compare_array_approx<T>(d2h_top.get(), h_top.get(), len, get_eps<T>()));

  // bprop
  test::normal_sync_cpu(h_top.get(), len, 0.f, 1.f, generator);
  softmax_fprop_cpu<T>(h_softmax_out.get(), h_bottom.get(), len, embedding_vector_size);
  core23::copy_sync(top_tensor.data(), h_top.get(), len * sizeof(T), top_tensor.device(),
                    core23::DeviceType::CPU);
  core23::copy_sync(softmax_layer.get_softmax_out_tensor().data(), h_softmax_out.get(),
                    len * sizeof(T), softmax_layer.get_softmax_out_tensor().device(),
                    core23::DeviceType::CPU);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  softmax_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  core23::copy_sync(d2h_bottom_grad.get(), bottom_tensor.data(), len * sizeof(T),
                    core23::DeviceType::CPU, bottom_tensor.device());
  softmax_bprop_cpu<T>(h_bottom_grad.get(), h_top.get(), h_softmax_out.get(), len,
                       embedding_vector_size);
  ASSERT_TRUE(
      test::compare_array_approx<T>(d2h_bottom_grad.get(), h_bottom_grad.get(), len, get_eps<T>()));
}

}  // namespace

TEST(softmax_layer, fp32_100x100) { softmax_test<float>(100, 100); }
TEST(softmax_layer, fp32_100x128) { softmax_test<float>(100, 128); }
TEST(softmax_layer, fp32_256x384) { softmax_test<float>(256, 384); }
TEST(softmax_layer, fp32_512x512) { softmax_test<float>(512, 512); }
TEST(softmax_layer, fp32_256x1024) { softmax_test<float>(256, 1024); }
TEST(softmax_layer, fp32_1024x512) { softmax_test<float>(1024, 512); }
TEST(softmax_layer, fp16_2x16) { softmax_test<__half>(2, 16); }
