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
#include <optimizers/adam_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

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

template <typename T>
AdamOptimizer<T>::AdamOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                             const std::shared_ptr<BufferBlock2<T>>& opt_buf,
                             const std::shared_ptr<GPUResource>& gpu_resource, float learning_rate,
                             float beta1, float beta2, float epsilon, float scaler)
    : Optimizer(weight_main, gpu_resource, learning_rate,
                scaler),
      wgrad_(wgrad),
      t_(0),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon) {
  if(weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    CK_THROW_(Error_t::WrongInput,
                  "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight_main.get_num_elements()}, &m_);
  opt_buf->reserve({weight_main.get_num_elements()}, &v_);
}

template <typename T>
void AdamOptimizer<T>::initialize() {
  CK_CUDA_THROW_(cudaMemsetAsync(m_.get_ptr(), 0, m_.get_size_in_bytes(), gpu_resource_->get_stream()));
  CK_CUDA_THROW_(cudaMemsetAsync(v_.get_ptr(), 0, v_.get_size_in_bytes(), gpu_resource_->get_stream()));
}

template <typename T>
void AdamOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  const size_t len = weight_main_.get_num_elements();
  constexpr size_t block_dim = 256;
  const size_t grid_dim = (len - 1) / block_dim + 1;

  ++t_;
  const float alpha_t = lr_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

  float* weight = weight_main_.get_ptr();

  T *m = m_.get_ptr();
  T *v = v_.get_ptr();
  const T *wgrad = wgrad_.get_ptr();
  adam_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, m, v, wgrad, alpha_t, beta1_, beta2_, epsilon_, scaler_);
#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}


template class AdamOptimizer<float>;
template class AdamOptimizer<__half>;

}  // namespace HugeCTR
