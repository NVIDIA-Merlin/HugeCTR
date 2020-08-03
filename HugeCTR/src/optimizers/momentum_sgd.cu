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

#include <optimizers/momentum_sgd.hpp>

namespace {

__device__ __forceinline__ void momentumSGD_update_device(
    float* weight_ptr, float* momentum_ptr, float wgrad,
    HugeCTR::MomentumSGDHyperParameters hyper_parameters, float scaler) {
  momentum_ptr[0] =
      hyper_parameters.momentum_factor * momentum_ptr[0] - hyper_parameters.lr * wgrad / scaler;
  weight_ptr[0] += momentum_ptr[0];

  // if(isnan(weight_ptr[0])){
  //   int count = atomicAdd(&debug_counter, 1);
  //   if(count < 10)
  //     printf("%f;", wgrad);
  // }

  return;
}

__global__ void momentumSGD_update_kernel(float* weight_ptr, float* momentum_ptr,
                                          const float* wgrad_ptr, int size,
                                          HugeCTR::MomentumSGDHyperParameters hyper_parameters,
                                          float scaler) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    momentumSGD_update_device(weight_ptr + idx, momentum_ptr + idx, wgrad_ptr[idx],
                              hyper_parameters, scaler);
  }
  return;
}

}  // namespace

namespace HugeCTR {

void MomentumSGD::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  constexpr int block_dim = 256;
  int grid_dim = (weight_->get_num_elements() + block_dim - 1) / block_dim;
  float* weight_ptr = weight_->get_ptr_with_offset(0);
  const float* wgrad_ptr = wgrad_->get_ptr_with_offset(0);
  float* momentum_ptr = momentum_->get_ptr_with_offset(0);

  MomentumSGDHyperParameters hyper_parameters = {lr_, momentum_factor_};
  momentumSGD_update_kernel<<<grid_dim, block_dim, 0, stream>>>(
      weight_ptr, momentum_ptr, wgrad_ptr, weight_->get_num_elements(), hyper_parameters, scaler_);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
