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

#include <general_buffer2.hpp>
#include <optimizers/adagrad_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {
namespace {

template <typename T>
__global__ void ada_grad_update_kernel(int len, float* weight, const T* wgrad, float* sum, float lr,
                                       const float epsilon, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float accum_ = sum[i];
    accum_ += gi * gi;
    float std_ = epsilon + sqrtf(accum_);
    weight[i] -= lr * gi / std_;
    sum[i] = accum_;
  }
}
}  // namespace

template <typename T>
AdaGradOptimizer<T>::AdaGradOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                                      const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                                      const std::shared_ptr<GPUResource>& gpu_resource,
                                      float learning_rate, float initial_accu_value, float epsilon,
                                      float scaler)
    : Optimizer(weight_main, gpu_resource, learning_rate, scaler),
      wgrad_(wgrad),
      initial_accumulator_value_(initial_accu_value),
      epsilon_(epsilon) {
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight_main.get_num_elements()}, &accum_);
}

template <typename T>
void AdaGradOptimizer<T>::initialize() {
  HCTR_LIB_THROW(cudaMemsetAsync(accum_.get_ptr(), initial_accumulator_value_,
                                 accum_.get_size_in_bytes(), gpu_resource_->get_stream()));
}

template <typename T>
void AdaGradOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_main_.get_ptr();
  const T* wgrad = wgrad_.get_ptr();
  ada_grad_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
      len, weight, wgrad, accum_.get_ptr(), lr_, epsilon_, scaler_);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class AdaGradOptimizer<float>;
template class AdaGradOptimizer<__half>;
}  // namespace HugeCTR
