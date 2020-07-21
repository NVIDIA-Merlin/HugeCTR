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
__global__ void adam_kernel(int len, float* weight_main, const T* wgrad,
                            T* m, T* v,
                            float alpha_t, float beta1, float beta2,
                            float epsilon, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = static_cast<float>(wgrad[i]) / scaler;
    float mi = beta1 * static_cast<float>(m[i]) + (1.f - beta1) * gi;
    float vi = beta2 * static_cast<float>(v[i]) + (1.f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    weight_main[i] -= alpha_t * mi / (sqrt(vi) + epsilon);
  }
}

}  // namespace

template <typename T>
AdamOptimizer<T>::AdamOptimizer(const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                                const std::shared_ptr<GeneralBuffer<T>>& wgrad,
                                const std::shared_ptr<GeneralBuffer<T>>& weight_sub,
                                int device_id,
                                float alpha,
                                float beta1, float beta2,
                                float epsilon, float scaler)
    : Optimizer(weight_main, device_id, alpha, scaler),
      m_(weight_main->get_num_elements(), device_id),
      v_(weight_main->get_num_elements(), device_id),
      t_(0),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      wgrad_(wgrad),
      weight_sub_(weight_sub) {
  m_.reset_sync();
  v_.reset_sync();
  if (weight_main_->get_num_elements() != wgrad_->get_num_elements()) {
    CK_THROW_(Error_t::WrongInput, "weight_main_ and wgrad_ have different lengths");
  }
}

template <typename T>
void AdamOptimizer<T>::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_main_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight_main = weight_main_->get_ptr_with_offset(0);
  const T* wgrad = wgrad_->get_ptr_with_offset(0);
  T* m = m_.get_ptr_with_offset(0);
  T* v = v_.get_ptr_with_offset(0);

  ++t_;
  const float alpha_t = lr_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));
  adam_kernel<<<grid_dim, block_dim, 0, stream>>>(len, weight_main, wgrad, m, v,
                                                  alpha_t, beta1_, beta2_,
                                                  epsilon_, scaler_);

  if (weight_sub_) {
    T* weight_sub = weight_sub_->get_ptr_with_offset(0);
    convert_array<<<grid_dim, block_dim, 0, stream>>>(weight_sub, weight_main, len);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class AdamOptimizer<float>;
template class AdamOptimizer<__half>;

}  // namespace HugeCTR
