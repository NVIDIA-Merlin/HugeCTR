/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "HugeCTR/include/optimizers/adam_optimizer.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {

template <typename T>
__global__ void adam_update_kernel(int len, float* weight, T* m, T* v, const T* wgrad,
                                   float alpha_t, float beta1, float beta2, float epsilon,
                                   float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float mi = beta1 * TypeConvertFunc<float, T>::convert(m[i]) + (1.f - beta1) * gi;
    float vi = beta2 * TypeConvertFunc<float, T>::convert(v[i]) + (1.f - beta2) * gi * gi;
    m[i] = TypeConvertFunc<T, float>::convert(mi);
    v[i] = TypeConvertFunc<T, float>::convert(vi);
    weight[i] -= alpha_t * mi / (sqrt(vi) + epsilon);
  }
}

}  // namespace

AdamOptimizer::AdamOptimizer(const GeneralBufferPtr<float>& weight_main,
                             const GeneralBufferPtr<float>& fp32_wgrad,
                             const GeneralBufferPtr<__half>& fp16_wgrad, bool mixed_precision,
                             int device_id, float learning_rate, float beta1, float beta2,
                             float epsilon, float scaler)
    : Optimizer(weight_main, fp32_wgrad, fp16_wgrad, mixed_precision, device_id, learning_rate,
                scaler),
      t_(0),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon) {
  if (mixed_precision) {
    fp16_m_.reserve(weight_main->get_num_elements());
    fp16_m_.init(device_id);
    fp16_m_.reset_sync();
    fp16_v_.reserve(weight_main->get_num_elements());
    fp16_v_.init(device_id);
    fp16_v_.reset_sync();
  } else {
    fp32_m_.reserve(weight_main->get_num_elements());
    fp32_m_.init(device_id);
    fp32_m_.reset_sync();
    fp32_v_.reserve(weight_main->get_num_elements());
    fp32_v_.init(device_id);
    fp32_v_.reset_sync();
  }
}

void AdamOptimizer::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const size_t len = weight_main_->get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  ++t_;
  const float alpha_t = lr_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

  float* weight = weight_main_->get_ptr_with_offset(0);

  if (mixed_precision_) {
    __half* fp16_m = fp16_m_.get_ptr_with_offset(0);
    __half* fp16_v = fp16_v_.get_ptr_with_offset(0);
    const __half* fp16_wgrad = fp16_wgrad_->get_ptr_with_offset(0);

    adam_update_kernel<<<grid_dim, block_dim, 0, stream>>>(
        len, weight, fp16_m, fp16_v, fp16_wgrad, alpha_t, beta1_, beta2_, epsilon_, scaler_);
  } else {
    float* fp32_m = fp32_m_.get_ptr_with_offset(0);
    float* fp32_v = fp32_v_.get_ptr_with_offset(0);
    const float* fp32_wgrad = fp32_wgrad_->get_ptr_with_offset(0);

    adam_update_kernel<<<grid_dim, block_dim, 0, stream>>>(
        len, weight, fp32_m, fp32_v, fp32_wgrad, alpha_t, beta1_, beta2_, epsilon_, scaler_);
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
