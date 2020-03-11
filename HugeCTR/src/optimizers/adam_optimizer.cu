/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace {

__global__ void adam_kernel(int len, float* weight, const float* wgrad, float* m, float* v,
                            float alpha_t, float beta1, float beta2, float epsilon) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int scaler = 1;
#ifdef SCALE_128
  scaler = 128;
#elif SCALE_256
  scaler = 256;
#elif SCALE_512
  scaler = 512;
#elif SCALE_1024
  scaler = 1024;
#else
  scaler = 1;
#endif
  if (i < len) {
    float gi = wgrad[i];
    float mi = beta1 * m[i] + (1 - beta1) * gi;
    float vi = beta2 * v[i] + (1 - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    weight[i] -= (double)alpha_t * mi / (sqrt(vi) + epsilon) / scaler;
  }
}

}  // namespace

namespace HugeCTR {

void AdamOptimizer::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  const int len = weight_->get_num_elements();
  const int block_dim = 256;
  const int grid_dim = (len - 1) / block_dim + 1;

  float* weight = weight_->get_ptr_with_offset(0);
  const float* wgrad = wgrad_->get_ptr_with_offset(0);
  float* m = m_.get_ptr_with_offset(0);
  float* v = v_.get_ptr_with_offset(0);

  ++t_;
  const float alpha_t = lr_ * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));
  adam_kernel<<<grid_dim, block_dim, 0, stream>>>(len, weight, wgrad, m, v, alpha_t, beta1_, beta2_,
                                                  epsilon_);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
