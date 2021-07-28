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
template <typename T>
MomentumSGDOptimizer<T>::MomentumSGDOptimizer(const Tensor2<float>& weight, const Tensor2<T>& wgrad,
                                              const std::shared_ptr<BufferBlock2<T>>& opt_buf,
                                              const std::shared_ptr<GPUResource>& gpu_resource,
                                              float learning_rate, float momentum_factor,
                                              float scaler)
    : Optimizer(weight, gpu_resource, learning_rate, scaler),
      wgrad_(wgrad),
      momentum_factor_(momentum_factor) {
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    CK_THROW_(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight.get_num_elements()}, &momentum_);
}

template <typename T>
void MomentumSGDOptimizer<T>::initialize() {
  CK_CUDA_THROW_(cudaMemsetAsync(momentum_.get_ptr(), 0, momentum_.get_size_in_bytes(),
                                 gpu_resource_->get_stream()));
}

template <typename T>
void MomentumSGDOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_main_.get_ptr();

  T* momentum = momentum_.get_ptr();
  T* wgrad = wgrad_.get_ptr();
  momentum_sgd_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
      len, weight, momentum, wgrad, lr_, momentum_factor_, scaler_);

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class MomentumSGDOptimizer<float>;
template class MomentumSGDOptimizer<__half>;

}  // namespace HugeCTR
