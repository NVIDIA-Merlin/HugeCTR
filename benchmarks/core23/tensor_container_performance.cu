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

#include <base/debug/logger.hpp>
#include <chrono>
#include <core23/cuda_primitives.cuh>
#include <core23/data_type.hpp>
#include <core23/data_type_helpers.cuh>
#include <core23/macros.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_container.hpp>
#include <cstdint>
#include <random>

#include "HugeCTR/core/tensor.hpp"

namespace {

using namespace HugeCTR::core23;

constexpr int NUM_ITER = 256;
constexpr int64_t TensorDims = 2;
constexpr int64_t ContainerDims = 1;
constexpr int64_t tensor_width = 1024;

__constant__ uint8_t workspace[16 * 8 * sizeof(TensorView<float, TensorDims>)];

// easier to use, better performance
template <typename T>
__global__ void update_kernel_flattened(TensorView<float, 1> weight, TensorView<T, 1> m,
                                        TensorView<T, TensorDims - 1> v,
                                        const TensorView<T, 1> wgrad, float alpha_t, float beta1,
                                        float beta2, float epsilon, float scalar) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto len = weight.size(0);
  if (i < len) {
    float gi = TypeConverter<float, T>::value(wgrad[i]) / scalar;
    float mi = beta1 * TypeConverter<float, T>::value(m[i]) + (1.f - beta1) * gi;
    float vi = beta2 * TypeConverter<float, T>::value(v[i]) + (1.f - beta2) * gi * gi;
    m[i] = TypeConverter<T, float>::value(mi);
    v[i] = TypeConverter<T, float>::value(vi);
    weight[i] -= alpha_t * mi / (sqrtf(vi) + epsilon);
  }
}

// harder to use, worse performance
template <typename T>
__global__ void update_kernel_full(
    TensorContainer<float, TensorDims, ContainerDims>::View weight_tensor_container,
    typename TensorContainer<T, TensorDims, ContainerDims>::View m_tensor_container,
    typename TensorContainer<T, TensorDims, ContainerDims>::View v_tensor_container,
    const typename TensorContainer<T, TensorDims, ContainerDims>::View wgrad_tensor_container,
    float alpha_t, float beta1, float beta2, float epsilon, float scalar) {
  const int64_t h_base = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t i_base = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_tensors = weight_tensor_container.size(0);
  for (int64_t h = h_base; h < num_tensors; h += blockDim.y * gridDim.y) {
    auto& weight = weight_tensor_container[h];
    auto& m = m_tensor_container[h];
    auto& v = v_tensor_container[h];
    const auto& wgrad = wgrad_tensor_container[h];
    const auto num_elements = weight.size(0) * weight.size(1);
    const __half2* wgrad2 = reinterpret_cast<__half2*>(&wgrad[0][0]);
    __half2* m2 = reinterpret_cast<__half2*>(&m[0][0]);
    __half2* v2 = reinterpret_cast<__half2*>(&v[0][0]);
    float2* weight2 = reinterpret_cast<float2*>(&weight[0][0]);
    for (int64_t i = i_base; i < num_elements / 2; i += blockDim.x * gridDim.x) {
      float2 gi = __half22float2(wgrad2[i]);
      gi.x = gi.x / scalar;
      gi.y = gi.y / scalar;
      float2 mi = __half22float2(m2[i]);
      mi.x = beta1 * mi.x + (1.f - beta1) * gi.x;
      mi.y = beta1 * mi.y + (1.f - beta1) * gi.y;
      float2 vi = __half22float2(v2[i]);
      vi.x = beta2 * vi.x + (1.f - beta2) * gi.x * gi.x;
      vi.y = beta2 * vi.y + (1.f - beta2) * gi.y * gi.y;
      m2[i] = __float22half2_rn(mi);
      float2 new_weight2 = weight2[i];
      new_weight2.x -= (alpha_t * mi.x / (sqrtf(vi.x) + epsilon));
      new_weight2.y -= (alpha_t * mi.y / (sqrtf(vi.y) + epsilon));
      v2[i] = __float22half2_rn(vi);
      weight2[i] = new_weight2;
    }
    int64_t i = i_base;
    if (i == 0 && num_elements % 2 > 0) {
      const __half* wgrad1 = &wgrad[0][0];
      __half* m1 = &m[0][0];
      __half* v1 = &v[0][0];
      float* weight1 = &weight[0][0];
      const float gi = TypeConverter<float, T>::value(wgrad1[i]) / scalar;
      const float mi = beta1 * TypeConverter<float, T>::value(m1[i]) + (1.f - beta1) * gi;
      const float vi = beta2 * TypeConverter<float, T>::value(v1[i]) + (1.f - beta2) * gi * gi;
      m1[i] = TypeConverter<T, float>::value(mi);
      v1[i] = TypeConverter<T, float>::value(vi);
      weight1[i] -= alpha_t * mi / (sqrtf(vi) + epsilon);
    }
  }
}

// baseline
template <typename T>
__global__ void update_kernel_base(int len, float* weight, T* m, T* v, const T* wgrad,
                                   float alpha_t, float beta1, float beta2, float epsilon,
                                   float scalar) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConverter<float, T>::value(wgrad[i]) / scalar;
    float mi = beta1 * TypeConverter<float, T>::value(m[i]) + (1.f - beta1) * gi;
    float vi = beta2 * TypeConverter<float, T>::value(v[i]) + (1.f - beta2) * gi * gi;
    m[i] = TypeConverter<T, float>::value(mi);
    v[i] = TypeConverter<T, float>::value(vi);
    weight[i] -= alpha_t * mi / (sqrtf(vi) + epsilon);
  }
}

template <typename T>
void initialize_container(TensorContainer<T, TensorDims, ContainerDims>& container) {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);

  for (int64_t t = 0; t < container.size(0); t++) {
    auto tensor = container[t];
    auto size = tensor.num_elements();
    std::vector<T> h_ins(size);
    for (int64_t i = 0; i < size; i++) {
      h_ins[i] = __float2half(uniform_dist(e));
    }
    copy_sync(container[t].data(), h_ins.data(), h_ins.size() * sizeof(T), container[t].device(),
              DeviceType::CPU);
  }
}

template <typename T>
void copy_container_to_tensor(Tensor& dst_tensor,
                              const TensorContainer<T, TensorDims, ContainerDims>& src_container) {
  int64_t offset = 0;
  for (int64_t t = 0; t < src_container.size(0); t++) {
    const auto src_tensor = src_container[t];
    copy_sync(dst_tensor.data<T>() + offset, src_tensor.data(), src_tensor.num_bytes(),
              dst_tensor.device(), src_tensor.device());
    offset += src_tensor.num_elements();
  }
}

template <typename T>
void tensor_container_performance(std::vector<Shape> shapes, bool flatten) {
  int64_t num_elements = 0;
  int64_t num_tensors = shapes.size();

  TensorParams common_tensor_params = TensorParams().data_type(ScalarType::Float);
  std::vector<Tensor> weight_tensors, m_tensors, v_tensors, wgrad_tensors;
  for (auto shape : shapes) {
    TensorParams tensor_params = common_tensor_params.shape(shape);
    BufferParams buffer_params;
    buffer_params.channel = std::string("WEIGHT_CONTAINER_BUFFER_");
    weight_tensors.emplace_back(tensor_params.buffer_params(buffer_params));

    buffer_params.channel = std::string("M_CONTAINER_BUFFER_");
    m_tensors.emplace_back(
        tensor_params.buffer_params(buffer_params).data_type(ToScalarType<T>::value));

    buffer_params.channel = std::string("V_CONTAINER_BUFFER_");
    v_tensors.emplace_back(
        tensor_params.buffer_params(buffer_params).data_type(ToScalarType<T>::value));

    buffer_params.channel = std::string("WGRAD_CONTAINER_BUFFER_");
    wgrad_tensors.emplace_back(
        tensor_params.buffer_params(buffer_params).data_type(ToScalarType<T>::value));

    num_elements += shape.size();
  }

  char* workspace_ptr_base = nullptr;
  HCTR_LIB_THROW(cudaGetSymbolAddress((void**)&workspace_ptr_base, workspace));
  char* workspace_ptr_0 = workspace_ptr_base;
  char* workspace_ptr_1 = workspace_ptr_0 + sizeof(TensorView<float, TensorDims>) * num_tensors;
  char* workspace_ptr_2 = workspace_ptr_1 + sizeof(TensorView<T, TensorDims>) * num_tensors;
  char* workspace_ptr_3 = workspace_ptr_2 + sizeof(TensorView<T, TensorDims>) * num_tensors;

  TensorContainer<float, TensorDims, ContainerDims> weight_tensor_container(
      workspace_ptr_0, std::move(weight_tensors), Shape({num_tensors}));
  TensorContainer<T, TensorDims, ContainerDims> m_tensor_container(
      workspace_ptr_1, std::move(m_tensors), Shape({num_tensors}));
  TensorContainer<T, TensorDims, ContainerDims> v_tensor_container(
      workspace_ptr_2, std::move(v_tensors), Shape({num_tensors}));
  TensorContainer<T, TensorDims, ContainerDims> wgrad_tensor_container(
      workspace_ptr_3, std::move(wgrad_tensors), Shape({num_tensors}));

  initialize_container(weight_tensor_container);
  initialize_container(m_tensor_container);
  initialize_container(v_tensor_container);
  initialize_container(wgrad_tensor_container);

  BufferParams buffer_params;
  buffer_params.channel = std::string("REF_TENSOR_BUFFER");
  TensorParams ref_tensor_params =
      common_tensor_params.shape({num_elements}).buffer_params(buffer_params);

  Tensor weight_ref_tensor(ref_tensor_params);
  Tensor m_ref_tensor(ref_tensor_params);
  Tensor v_ref_tensor(ref_tensor_params);
  Tensor wgrad_ref_tensor(ref_tensor_params);
  copy_container_to_tensor(weight_ref_tensor, weight_tensor_container);
  copy_container_to_tensor(m_ref_tensor, m_tensor_container);
  copy_container_to_tensor(v_ref_tensor, v_tensor_container);
  copy_container_to_tensor(wgrad_ref_tensor, wgrad_tensor_container);

  CUDAStream stream(cudaStreamDefault, 0);

  auto b_t = std::chrono::steady_clock::now();
  if (flatten) {
    auto weight_tensor_container_flattened = weight_tensor_container.flatten();
    auto m_tensor_container_flattened = m_tensor_container.flatten();
    auto v_tensor_container_flattened = v_tensor_container.flatten();
    auto wgrad_tensor_container_flattened = wgrad_tensor_container.flatten();

    for (int iter = 0; iter < NUM_ITER; iter++) {
      const size_t len = num_elements;
      constexpr size_t block = 256;
      const size_t grid = (len - 1) / block + 1;
      update_kernel_flattened<T><<<grid, block, 0, stream()>>>(
          weight_tensor_container_flattened, m_tensor_container_flattened,
          v_tensor_container_flattened, wgrad_tensor_container_flattened, 0.1f, 0.1f, 0.1f, 0.1f,
          1024.0f);
    }
  } else {
    auto weight_tensor_container_view = weight_tensor_container.view();
    auto m_tensor_container_view = m_tensor_container.view();
    auto v_tensor_container_view = v_tensor_container.view();
    auto wgrad_tensor_container_view = wgrad_tensor_container.view();
    for (int iter = 0; iter < NUM_ITER; iter++) {
      dim3 block(256, 1, 1);
      dim3 grid(((num_elements + num_tensors - 1) / num_tensors + block.x - 1) / block.x / 3,
                num_tensors / 2);
      update_kernel_full<T><<<grid, block, 0, stream()>>>(
          weight_tensor_container_view, m_tensor_container_view, v_tensor_container_view,
          wgrad_tensor_container_view, 0.1f, 0.1f, 0.1f, 0.1f, 1024.0f);
    }
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  auto e_t = std::chrono::steady_clock::now();
  HCTR_LOG_S(INFO, ROOT)
      << std::chrono::duration_cast<std::chrono::nanoseconds>(e_t - b_t).count() / NUM_ITER
      << " ns to run update_kernel() with "
      << "TensorContainer(flatten = " << std::boolalpha << flatten << ")" << std::endl;

  b_t = std::chrono::steady_clock::now();
  auto weight_ref = weight_ref_tensor.data<float>();
  auto m_ref = m_ref_tensor.data<T>();
  auto v_ref = v_ref_tensor.data<T>();
  auto wgrad_ref = wgrad_ref_tensor.data<T>();
  for (int iter = 0; iter < NUM_ITER; iter++) {
    const size_t len = num_elements;
    constexpr size_t block = 256;
    const size_t grid = (len - 1) / block + 1;
    update_kernel_base<T><<<grid, block, 0, stream()>>>(len, weight_ref, m_ref, v_ref, wgrad_ref,
                                                        0.1f, 0.1f, 0.1f, 0.1f, 1024.0f);
  }
  HCTR_LIB_THROW(cudaStreamSynchronize(stream()));
  e_t = std::chrono::steady_clock::now();
  HCTR_LOG_S(INFO, ROOT)
      << std::chrono::duration_cast<std::chrono::nanoseconds>(e_t - b_t).count() / NUM_ITER
      << " ns to run update_kernel() with Tensor " << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool flatten = true;
    if (argc >= 2) {
      std::istringstream(std::string(argv[1])) >> flatten;
    }

    tensor_container_performance<__half>({{13, 512},
                                          {1, 512},
                                          {512, 256},
                                          {1, 256},
                                          {256, 128},
                                          {1, 128},
                                          {479, 1024},
                                          {1, 1024},
                                          {1024, 1024},
                                          {1, 1024},
                                          {1024, 512},
                                          {1, 512},
                                          {512, 256},
                                          {1, 256},
                                          {256, 1},
                                          {1, 2}},
                                         flatten);
  } catch (...) {
    HCTR_LOG_S(INFO, ROOT) << "Something is wrong" << std::endl;
  }
}