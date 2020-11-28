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

#include <optimizers/sgd_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void sgd_update_kernel(int len, float* weight, const T* wgrad, float lr, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    weight[i] -= lr * gi;
  }
}

}  // namespace

SGDOptimizer::SGDOptimizer(const Tensor2<float>& weight_main, const Tensor2<float>& fp32_wgrad,
                           const Tensor2<__half>& fp16_wgrad, bool mixed_precision,
                           const std::shared_ptr<GPUResource>& gpu_resource, float lr, float scaler)
    : Optimizer(weight_main, fp32_wgrad, fp16_wgrad, mixed_precision, gpu_resource, lr, scaler) {}

void SGDOptimizer::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_main_.get_ptr();

  if (mixed_precision_) {
    const __half* fp16_wgrad = fp16_wgrad_.get_ptr();
    sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp16_wgrad, lr_, scaler_);
  } else {
    const float* fp32_wgrad = fp32_wgrad_.get_ptr();
    sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, fp32_wgrad, lr_, scaler_);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
