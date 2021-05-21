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
__device__ inline void sgd_update_device(int len, float* weight, const T* wgrad, float lr, float scaler) {
  constexpr int vec_width = sizeof(float4)/sizeof(float);
  using T4 = typename std::conditional<(sizeof(T) == 4),float4,float2>::type;

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (vec_width*gid + (vec_width-1) < len &&
     (intptr_t)weight % sizeof(float4) == 0 &&
     (intptr_t)wgrad  % sizeof(T4)     == 0) {

    float4   weight4 = reinterpret_cast<float4*>  (weight)[gid];
    const T4 wgrad4  = reinterpret_cast<const T4*>(wgrad) [gid];

    float*   weight_vec = reinterpret_cast<float*>  (&weight4);
    const T* wgrad_vec  = reinterpret_cast<const T*>(&wgrad4);

    #pragma unroll vec_width
    for (int i=0; i<vec_width; i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad_vec[i]) / scaler;
      weight_vec[i] -= lr * gi;
    }

    reinterpret_cast<float4*>(weight)[gid] = weight4;
    
  } else {
    #pragma unroll vec_width
    for (int i=vec_width*gid; i<min(len, vec_width*(gid+1)); i++) {
      float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
      weight[i] -= lr * gi;
    }
  }
}

template <typename T>
__global__ void sgd_update_kernel(int len, float* weight, const T* wgrad, float lr, float scaler) {
  sgd_update_device(len, weight, wgrad, lr, scaler);
}

template <typename T>
__global__ void sgd_update_kernel(int len, float* weight, const T* wgrad,
                                  const float* lr_ptr, float scaler) {
  sgd_update_device(len, weight, wgrad, *lr_ptr, scaler);
}

}  // namespace

SGDOptimizer::SGDOptimizer(const Tensor2<float>& weight_main, const Tensor2<float>& fp32_wgrad,
                           const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
                           const std::shared_ptr<GPUResource>& gpu_resource, float lr, float scaler)
    : Optimizer(weight_main, fp32_wgrad, fp16_wgrad, mixed_precision, gpu_resource, lr, scaler) {}

void SGDOptimizer::update() {
  CudaDeviceContext context(get_device_id());
  PROFILE_RECORD("update.start", gpu_resource_->get_stream(), false);
  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  constexpr int vec_width = sizeof(float4) / sizeof(float);
  const size_t grid_dim = (len + block_dim*vec_width - 1) / (block_dim*vec_width);

  float* weight = weight_main_.get_ptr();

  if (gpu_learning_rate_scheduler_ == nullptr) {
    if (mixed_precision_) {
      const __half* fp16_wgrad = fp16_wgrad_.get_ptr();
      sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
          len, weight, fp16_wgrad, lr_, scaler_);
    } else {
      const float* fp32_wgrad = fp32_wgrad_.get_ptr();
      sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
          len, weight, fp32_wgrad, lr_, scaler_);
    }
  }
  else {
    float* lr_ptr = gpu_learning_rate_scheduler_->get_learning_rate();
    if (mixed_precision_) {
      const __half* fp16_wgrad = fp16_wgrad_.get_ptr();
      sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
          len, weight, fp16_wgrad, lr_ptr, scaler_);
    } else {
      const float* fp32_wgrad = fp32_wgrad_.get_ptr();
      sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
          len, weight, fp32_wgrad, lr_ptr, scaler_);
    }
    gpu_learning_rate_scheduler_->update();
  }

  PROFILE_RECORD("update.stop", gpu_resource_->get_stream(), false);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
