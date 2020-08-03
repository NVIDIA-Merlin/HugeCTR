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

#include <optimizers/sgd_optimizer_half.hpp>

namespace {

__global__ void sgd_kernel_half(int len, float* weight, const __half* wgrad, __half* weight_half,
                                float lr, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = (float)wgrad[i] / scaler;
    weight[i] -= lr * gi;
    weight_half[i] = weight[i];
  }
}

}  // namespace

namespace HugeCTR {

void SgdOptimizerHalf::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_->get_ptr_with_offset(0);
  const __half* wgrad = wgrad_->get_ptr_with_offset(0);
  __half* weight_half = weight_half_->get_ptr_with_offset(0);

  sgd_kernel_half<<<grid_dim, block_dim, 0, stream>>>(len, weight, wgrad, weight_half, lr_,
                                                      scaler_);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
