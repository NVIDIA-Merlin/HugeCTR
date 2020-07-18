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

namespace {

template <typename T>
__global__ void sgd_kernel(int len, float* weight, const T* wgrad, T* weight_tmp,
                           float lr, float scaler); 

template <>
__global__ void sgd_kernel<float>(int len, float* weight, const float* wgrad, float* weight_tmp,
                                  float lr, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = (float)wgrad[i] / scaler;
    weight[i] -= lr * gi;
  }
}

template <>
__global__ void sgd_kernel<__half>(int len, float* weight, const __half* wgrad, __half* weight_tmp,
                                   float lr, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = (float)wgrad[i] / scaler;
    weight[i] -= lr * gi;
    weight_tmp[i] = weight[i];
  }
}

}  // namespace

namespace HugeCTR {

template <typename T>
void SgdOptimizer<T>::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_->get_ptr_with_offset(0);
  const T* wgrad = wgrad_->get_ptr_with_offset(0);

  if (std::is_same<T, __half>::value) {
    T* weight_tmp = weight_tmp_->get_ptr_with_offset(0);
    sgd_kernel<T><<<grid_dim, block_dim, 0, stream>>>(len, weight, wgrad, weight_tmp, lr_,
                                                        scaler_);
  } else {
    T* weight_tmp = nullptr; 
    sgd_kernel<T><<<grid_dim, block_dim, 0, stream>>>(len, weight, wgrad, weight_tmp, lr_,
                                                        scaler_);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class SgdOptimizer<float>;
template class SgdOptimizer<__half>;

}  // namespace HugeCTR

