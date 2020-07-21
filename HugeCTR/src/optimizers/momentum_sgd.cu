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

#include "HugeCTR/include/optimizers/momentum_sgd.hpp"

#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {

__forceinline__
__device__ void momentumSGD_update_device(float* weight_main, float* momuntum,
                                          float wgrad,
                                          MomentumSGDHyperParameters hyper_parameters,
                                          float scaler) {
  float mv = hyper_parameters.momentum_factor * static_cast<float>(momuntum[0]) - 
             hyper_parameters.lr * wgrad / scaler;
  momuntum[0] = mv;
  weight_main[0] += mv;
  return;
}

template <typename T>
__global__ void momentumSGD_update_kernel(float* weight_main, float* momuntum,
                                          const T* wgrad, int size,
                                          MomentumSGDHyperParameters hyper_parameters,
                                          float scaler) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    momentumSGD_update_device(weight_main + idx, momuntum + idx, wgrad[idx],
                              hyper_parameters, scaler);
  }
  return;
}

}  // namespace

template <typename T>
MomentumSGD<T>::MomentumSGD(const std::shared_ptr<GeneralBuffer<float>>& weight_main,
                            const std::shared_ptr<GeneralBuffer<T>>& wgrad,
                            const std::shared_ptr<GeneralBuffer<T>>& weight_sub,
                            int device_id,
                            float learning_rate, float momentum_factor, float scaler)
    : Optimizer(weight_main, device_id, learning_rate, scaler),
      momentum_factor_(momentum_factor),
      wgrad_(wgrad),
      weight_sub_(weight_sub) {
  momentum_.reset(new GeneralBuffer<float>(weight_main_->get_num_elements(), device_id_));
  momentum_->reset_sync();
  if (weight_main_->get_num_elements() != wgrad_->get_num_elements()) {
    CK_THROW_(Error_t::WrongInput, "weight_main_ and wgrad_ have different lengths");
  }
}

template <typename T>
void MomentumSGD<T>::update(cudaStream_t stream) {
  CudaDeviceContext context(device_id_);

  constexpr int block_dim = 256;
  int grid_dim = (weight_main_->get_num_elements() + block_dim - 1) / block_dim;
  float* weight_main = weight_main_->get_ptr_with_offset(0);
  const T* wgrad = wgrad_->get_ptr_with_offset(0);
  float* momuntum = momentum_->get_ptr_with_offset(0);
  const int len = weight_main_->get_num_elements();

  MomentumSGDHyperParameters hyper_parameters = {lr_, momentum_factor_};
  momentumSGD_update_kernel<<<grid_dim, block_dim, 0, stream>>>(
      weight_main, momuntum, wgrad, len, hyper_parameters, scaler_);

  if (weight_sub_) {
    T* weight_sub = weight_sub_->get_ptr_with_offset(0);
    convert_array<<<grid_dim, block_dim, 0, stream>>>(weight_sub, weight_main, len);
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class MomentumSGD<float>;
template class MomentumSGD<__half>;

}  // namespace HugeCTR
