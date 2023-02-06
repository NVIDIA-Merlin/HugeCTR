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
#include <optimizers/ftrl_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void ftrl_update_kernel(int len, float* weight, float* z, float* n, const T* wgrad,
                                   float alpha, float beta, float lambda1, float lambda2,
                                   float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float ni_new = n[i] + gi * gi;
    float zi = z[i] + gi + (sqrt(n[i]) - sqrt(ni_new)) * weight[i] / alpha;
    float x = lambda1 * (1.0f - 2.0f * signbit(zi)) - zi;
    float y = sqrt(ni_new) / alpha + lambda2;
    n[i] = ni_new;
    z[i] = zi;
    weight[i] = x / y * signbit(lambda1 - abs(zi));
  }
}

}  // namespace

template <typename T>
FtrlOptimizer<T>::FtrlOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                                const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                                const std::shared_ptr<GPUResource>& gpu_resource,
                                float learning_rate, float beta, float lambda1, float lambda2,
                                float scaler)
    : Optimizer(weight_main, gpu_resource, learning_rate, scaler),
      wgrad_(wgrad),
      beta_(beta),
      lambda1_(lambda1),
      lambda2_(lambda2) {
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight_main.get_num_elements()}, &n_);
  opt_buf->reserve({weight_main.get_num_elements()}, &z_);
}

template <typename T>
void FtrlOptimizer<T>::initialize() {
  HCTR_LIB_THROW(
      cudaMemsetAsync(n_.get_ptr(), 0, n_.get_size_in_bytes(), gpu_resource_->get_stream()));
  HCTR_LIB_THROW(
      cudaMemsetAsync(z_.get_ptr(), 0, z_.get_size_in_bytes(), gpu_resource_->get_stream()));
}

template <typename T>
void FtrlOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  // const float alpha_t = lr_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

  float* weight = weight_main_.get_ptr();

  float* z = z_.get_ptr();
  float* n = n_.get_ptr();
  const T* wgrad = wgrad_.get_ptr();
  ftrl_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
      len, weight, z, n, wgrad, lr_, beta_, lambda1_, lambda2_ + beta_ / lr_, scaler_);
#ifndef NDEBUG
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class FtrlOptimizer<float>;
template class FtrlOptimizer<__half>;

}  // namespace HugeCTR
