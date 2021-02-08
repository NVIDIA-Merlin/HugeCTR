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

#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/optimizers/momentum_sgd_optimizer.hpp"
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace {

template <typename T>
__global__ void momentum_sgd_update_kernel(int len, float* weight, T* momentum, const T* wgrad,
                                           float lr, float momentum_factor, float scaler) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len) {
    float mv = momentum_factor * TypeConvertFunc<float, T>::convert(momentum[idx]) -
               lr * TypeConvertFunc<float, T>::convert(wgrad[idx]) / scaler;
    momentum[idx] = TypeConvertFunc<T, float>::convert(mv);
    weight[idx] += mv;
  }
  return;
}

}  // namespace

MomentumSGDOptimizer::MomentumSGDOptimizer(
    const Tensor2<float>& weight, const Tensor2<float>& fp32_wgrad,
    const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
    const std::shared_ptr<BufferBlock2<float>>& opt_buf,
    const std::shared_ptr<BufferBlock2<__half>>& opt_buf_half,
    const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate, float momentum_factor,
    float scaler)
    : Optimizer(weight, fp32_wgrad, fp16_wgrad, mixed_precision, gpu_resource, learning_rate,
                scaler),
      momentum_factor_(momentum_factor) {
  if (mixed_precision) {
    opt_buf_half->reserve({weight.get_num_elements()}, &fp16_momentum_);
  } else {
    opt_buf->reserve({weight.get_num_elements()}, &fp32_momentum_);
  }
}

void MomentumSGDOptimizer::initialize() {
  if (mixed_precision_) {
    CK_CUDA_THROW_(cudaMemsetAsync(fp16_momentum_.get_ptr(), 0, fp16_momentum_.get_size_in_bytes(),
                    gpu_resource_->get_stream()));
  } else {
    CK_CUDA_THROW_(cudaMemsetAsync(fp32_momentum_.get_ptr(), 0, fp32_momentum_.get_size_in_bytes(),
                    gpu_resource_->get_stream()));
  }
}

void MomentumSGDOptimizer::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_main_.get_ptr();

  if (mixed_precision_) {
    __half* fp16_momentum = fp16_momentum_.get_ptr();
    const __half* fp16_wgrad = fp16_wgrad_.get_ptr();

    momentum_sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp16_momentum, fp16_wgrad, lr_, momentum_factor_, scaler_);
  } else {
    float* fp32_momentum = fp32_momentum_.get_ptr();
    const float* fp32_wgrad = fp32_wgrad_.get_ptr();

    momentum_sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp32_momentum, fp32_wgrad, lr_, momentum_factor_, scaler_);
  }

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
