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

#include <general_buffer2.hpp>
#include <optimizers/nesterov_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void nesterov_update_kernel(int len, float* weight, T* accum, const T* wgrad, float lr,
                                       float mu, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float accum_old = TypeConvertFunc<float, T>::convert(accum[i]);
    float accum_new = mu * accum_old - lr * TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    accum[i] = TypeConvertFunc<T, float>::convert(accum_new);
    weight[i] += (-mu * accum_old + (1.f + mu) * accum_new);
  }
}

}  // namespace

NesterovOptimizer::NesterovOptimizer(const Tensor2<float>& weight_main,
                                     const Tensor2<float>& fp32_wgrad,
                                     const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
                                     const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& buff,
                                     const std::shared_ptr<GPUResource>& gpu_resource,
                                     float learning_rate, float momentum_factor, float scaler)
    : Optimizer(weight_main, fp32_wgrad, fp16_wgrad, mixed_precision, gpu_resource, learning_rate,
                scaler),
      mu_(momentum_factor) {
  if (mixed_precision) {
    buff->reserve({weight_main.get_num_elements()}, &fp16_accum_);
  } else {
    buff->reserve({weight_main.get_num_elements()}, &fp32_accum_);
  }
}

void NesterovOptimizer::initialize() {
  if (mixed_precision_) {
    cudaMemsetAsync(fp16_accum_.get_ptr(), 0, fp16_accum_.get_size_in_bytes(),
                    gpu_resource_->get_stream());
  } else {
    cudaMemsetAsync(fp32_accum_.get_ptr(), 0, fp32_accum_.get_size_in_bytes(),
                    gpu_resource_->get_stream());
  }
}

void NesterovOptimizer::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_main_.get_ptr();

  if (mixed_precision_) {
    __half* fp16_accum = fp16_accum_.get_ptr();
    const __half* fp16_wgrad = fp16_wgrad_.get_ptr();

    nesterov_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp16_accum, fp16_wgrad, lr_, mu_, scaler_);
  } else {
    float* fp32_accum = fp32_accum_.get_ptr();
    const float* fp32_wgrad = fp32_wgrad_.get_ptr();

    nesterov_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp32_accum, fp32_wgrad, lr_, mu_, scaler_);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
