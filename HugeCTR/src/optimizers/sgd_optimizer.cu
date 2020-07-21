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

#include "HugeCTR/include/optimizers/sgd_optimizer.hpp"

#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {

template <typename T>
__global__ void sgd_kernel(int len, float* weight_main, const T* wgrad, float lr,
                           float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = (float)wgrad[i] / scaler;
    weight_main[i] -= lr * gi;
  }
}

}  // namespace

template <typename T>
SgdOptimizer<T>::SgdOptimizer(const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                              const std::shared_ptr<GeneralBuffer<T>>& wgrad,
                              const std::shared_ptr<GeneralBuffer<T>>& weight_sub,
                              int device_id,
                              float lr, float scaler)
    : Optimizer(weight_main, device_id, lr, scaler),
      wgrad_(wgrad),
      weight_sub_(weight_sub) {

  if (weight_sub_ != nullptr &&
      wgrad_->get_num_elements() != weight_sub_->get_num_elements()) {
    CK_THROW_(Error_t::WrongInput,
              "wgrad_->get_num_elements() != weight_sub_->get_num_elements()");
  }
  if (weight_main_->get_num_elements() != wgrad_->get_num_elements()) {
    CK_THROW_(Error_t::WrongInput,
              "weight_main_.get_num_elements() != wgrad_.get_num_elements()");
  }
}

template <typename T>
void SgdOptimizer<T>::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_main_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight_main = weight_main_->get_ptr_with_offset(0);
  const T* wgrad = wgrad_->get_ptr_with_offset(0);

  sgd_kernel<T><<<grid_dim, block_dim, 0, stream>>>(len, weight_main, wgrad, lr_, scaler_);

  if (weight_sub_) {
    T* weight_sub = weight_sub_->get_ptr_with_offset(0);
    convert_array<<<grid_dim, block_dim, 0, stream>>>(weight_sub, weight_main, len);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class SgdOptimizer<float>;
template class SgdOptimizer<__half>;

}  // namespace HugeCTR

