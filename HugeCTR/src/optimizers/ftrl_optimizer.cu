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
#include <float.h>

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
    float zi =
        z[i] + gi + (sqrtf(n[i] + FLT_EPSILON) - sqrtf(ni_new + FLT_EPSILON)) * weight[i] / alpha;
    float x = lambda1 * (1.0f - 2.0f * signbit(zi)) - zi;
    float y = sqrtf(ni_new + FLT_EPSILON) / alpha + lambda2;
    n[i] = ni_new;
    z[i] = zi;
    weight[i] = x / y * signbit(lambda1 - abs(zi));
  }
}

}  // namespace

template <typename T>
FtrlOptimizer<T>::FtrlOptimizer(std::optional<WeightTensors> weight_tensors,
                                std::optional<WgradTensors<T>> wgrad_tensors,
                                const std::shared_ptr<GPUResource>& gpu_resource,
                                float learning_rate, float beta, float lambda1, float lambda2,
                                float scaler)
    : Optimizer(weight_tensors, gpu_resource, learning_rate, scaler),
      wgrad_tensors_(wgrad_tensors),
      beta_(beta),
      lambda1_(lambda1),
      lambda2_(lambda2) {
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(core23::Device(core23::DeviceType::GPU, gpu_resource->get_device_id()))
          .data_type(core23::ScalarType::Float)
          .shape(core23::Shape({weight_tensors_->flatten().size(0)}))
          .buffer_channel(GetOptStateBufferChannnel());

  z_tensor_ = core23::Tensor(tensor_params);
  n_tensor_ = core23::Tensor(tensor_params);
}
template <typename T>
void FtrlOptimizer<T>::initialize() {
  HCTR_LIB_THROW(
      cudaMemsetAsync(n_tensor_.data(), 0, n_tensor_.num_bytes(), gpu_resource_->get_stream()));
  HCTR_LIB_THROW(
      cudaMemsetAsync(z_tensor_.data(), 0, z_tensor_.num_bytes(), gpu_resource_->get_stream()));
}

template <typename T>
void FtrlOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  constexpr size_t block_dim = 256;

  auto flat_weight_tensor = weight_tensors_->flatten();
  auto flat_wgrad_tensor = wgrad_tensors_->flatten();
  float* weight = flat_weight_tensor.data();
  const T* wgrad = flat_wgrad_tensor.data();

  auto len = flat_weight_tensor.size(0);
  const size_t grid_dim = (len - 1) / block_dim + 1;

  float* z = z_tensor_.data<float>();
  float* n = n_tensor_.data<float>();
  ftrl_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
      len, weight, z, n, wgrad, lr_, beta_, lambda1_, lambda2_ + beta_ / lr_, scaler_);
}

template class FtrlOptimizer<float>;
template class FtrlOptimizer<__half>;

}  // namespace HugeCTR
