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

#include <optimizers/sgd_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__device__ inline void sgd_update_device(int len, float* weight, const T* wgrad, float lr,
                                         float scaler) {
  constexpr int vec_width = sizeof(float4) / sizeof(float);
  using T4 = typename std::conditional<(sizeof(T) == 4), float4, float2>::type;

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (vec_width * gid + (vec_width - 1) < len && (intptr_t)weight % sizeof(float4) == 0 &&
      (intptr_t)wgrad % sizeof(T4) == 0) {
    float4 weight4 = reinterpret_cast<float4*>(weight)[gid];
    const T4 wgrad4 = reinterpret_cast<const T4*>(wgrad)[gid];

    float* weight_vec = reinterpret_cast<float*>(&weight4);
    const T* wgrad_vec = reinterpret_cast<const T*>(&wgrad4);

#pragma unroll vec_width
    for (int i = 0; i < vec_width; i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad_vec[i]) / scaler;
      weight_vec[i] -= lr * gi;
    }

    reinterpret_cast<float4*>(weight)[gid] = weight4;

  } else {
#pragma unroll vec_width
    for (int i = vec_width * gid; i < min(len, vec_width * (gid + 1)); i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
      weight[i] -= lr * gi;
    }
  }
}

template <typename T>
__device__ inline void sgd_update_device(int len, float* weight, __half* weight_half,
                                         const T* wgrad, float lr, float scaler) {
  constexpr int vec_width = sizeof(float4) / sizeof(float);
  using T4 = typename std::conditional<(sizeof(T) == 4), float4, float2>::type;

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (vec_width * gid + (vec_width - 1) < len && (intptr_t)weight % sizeof(float4) == 0 &&
      (intptr_t)wgrad % sizeof(T4) == 0) {
    float4 weight4 = reinterpret_cast<float4*>(weight)[gid];
    float2 weight_half4 = reinterpret_cast<float2*>(weight_half)[gid];
    const T4 wgrad4 = reinterpret_cast<const T4*>(wgrad)[gid];

    float* weight_vec = reinterpret_cast<float*>(&weight4);
    __half* weight_half_vec = reinterpret_cast<__half*>(&weight_half4);
    const T* wgrad_vec = reinterpret_cast<const T*>(&wgrad4);

#pragma unroll vec_width
    for (int i = 0; i < vec_width; i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad_vec[i]) / scaler;
      weight_vec[i] -= lr * gi;
      weight_half_vec[i] = (__half)weight_vec[i];
    }

    reinterpret_cast<float4*>(weight)[gid] = weight4;
    reinterpret_cast<float2*>(weight_half)[gid] = weight_half4;

  } else {
#pragma unroll vec_width
    for (int i = vec_width * gid; i < min(len, vec_width * (gid + 1)); i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
      weight[i] -= lr * gi;
      weight_half[i] = (__half)weight[i];
    }
  }
}

template <typename T>
__global__ void sgd_update_kernel(int len, float* weight, __half* weight_half, const T* wgrad,
                                  float lr, float scaler, bool use_mixed_precision) {
  if (true == use_mixed_precision) {
    sgd_update_device(len, weight, weight_half, wgrad, lr, scaler);
  } else {
    sgd_update_device(len, weight, wgrad, lr, scaler);
  }
}

template <typename T>
__global__ void sgd_update_kernel(int len, float* weight, __half* weight_half, const T* wgrad,
                                  const float* lr_ptr, float scaler, bool use_mixed_precision) {
  if (true == use_mixed_precision) {
    sgd_update_device(len, weight, weight_half, wgrad, *lr_ptr, scaler);
  } else {
    sgd_update_device(len, weight, wgrad, *lr_ptr, scaler);
  }
}

}  // namespace

template <typename T>
SGDOptimizer<T>::SGDOptimizer(const Tensor2<float>& weight_main,
                              const Tensor2<__half>& weight_main_half, const Tensor2<T>& wgrad,
                              const std::shared_ptr<GPUResource>& gpu_resource, float lr,
                              float scaler, bool use_mixed_precision)
    : Optimizer(weight_main, gpu_resource, lr, scaler),
      wgrad_(wgrad),
      weight_main_half_(weight_main_half),
      use_mixed_precision_(use_mixed_precision) {
  optimizer_type_ = Optimizer_t::SGD;
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
}

template <typename T>
void SGDOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());
  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  constexpr int vec_width = sizeof(float4) / sizeof(float);
  const size_t grid_dim = (len + block_dim * vec_width - 1) / (block_dim * vec_width);

  float* weight = weight_main_.get_ptr();
  __half* weight_half = weight_main_half_.get_ptr();
  const T* wgrad = wgrad_.get_ptr();

  if (gpu_learning_rate_scheduler_ == nullptr) {
    sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, weight_half, wgrad, lr_, scaler_, use_mixed_precision_);
  } else {
    float* lr_ptr = gpu_learning_rate_scheduler_->get_learning_rate();
    sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, weight_half, wgrad, lr_ptr, scaler_, use_mixed_precision_);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class SGDOptimizer<float>;
template class SGDOptimizer<__half>;

}  // namespace HugeCTR
