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

#include <general_buffer2.hpp>
#include <optimizers/adagrad_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {
namespace {

template <typename T>
__global__ void ada_grad_update_kernel(int len, float* weight, const T* wgrad, float* sum, float lr,
                                       const float epsilon, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float accum_ = sum[i];
    accum_ += gi * gi;
    float std_ = epsilon + sqrtf(accum_);
    weight[i] -= lr * gi / std_;
    sum[i] = accum_;
  }
}
}  // namespace

template <typename T>
AdaGradOptimizer<T>::AdaGradOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                                      const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                                      const std::shared_ptr<GPUResource>& gpu_resource,
                                      float learning_rate, float initial_accu_value, float epsilon,
                                      float scaler)
    : Optimizer(weight_main, gpu_resource, learning_rate, scaler),
      wgrad_(wgrad),
      wgrad_tensors_({}),
      initial_accumulator_value_(initial_accu_value),
      epsilon_(epsilon) {
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight_main.get_num_elements()}, &accum_);
}

template <typename T>
AdaGradOptimizer<T>::AdaGradOptimizer(std::vector<core23::Tensor> weight_tensors,
                                      std::vector<core23::Tensor> wgrad_tensors,
                                      const std::shared_ptr<GPUResource>& gpu_resource,
                                      float learning_rate, float initial_accu_value, float epsilon,
                                      float scaler)
    : Optimizer(weight_tensors, gpu_resource, learning_rate, scaler),
      wgrad_tensors_(std::make_optional<WgradTensors<T>>(
          std::move(wgrad_tensors), core23::Shape({static_cast<int64_t>(wgrad_tensors.size())}))),
      initial_accumulator_value_(initial_accu_value),
      epsilon_(epsilon) {
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(core23::Device(core23::DeviceType::GPU, gpu_resource->get_device_id()))
          .data_type(core23::ScalarType::Float)
          .shape(core23::Shape({weight_tensors_->flatten().size(0)}))
          .buffer_channel(GetOptStateBufferChannnel());

  accum_tensor_ = core23::Tensor(tensor_params);
}

template <typename T>
void AdaGradOptimizer<T>::initialize() {
  if (!wgrad_tensors_) {
    HCTR_LIB_THROW(cudaMemsetAsync(accum_.get_ptr(), initial_accumulator_value_,
                                   accum_.get_size_in_bytes(), gpu_resource_->get_stream()));
  } else {
    HCTR_LIB_THROW(cudaMemsetAsync(accum_tensor_.data(), initial_accumulator_value_,
                                   accum_tensor_.num_bytes(), gpu_resource_->get_stream()));
  }
}

template <typename T>
void AdaGradOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  constexpr size_t block_dim = 256;

  if (!wgrad_tensors_) {
    const size_t len = weight_main_.get_num_elements();
    float* weight = weight_main_.get_ptr();
    const T* wgrad = wgrad_.get_ptr();
    float* accum = accum_.get_ptr();
    const size_t grid_dim = (len - 1) / block_dim + 1;
    ada_grad_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, wgrad, accum, lr_, epsilon_, scaler_);
  } else {
    auto flat_weight_tensor = weight_tensors_->flatten();
    auto flat_wgrad_tensor = wgrad_tensors_->flatten();
    float* weight = flat_weight_tensor.data();
    const T* wgrad = flat_wgrad_tensor.data();
    auto len = flat_weight_tensor.size(0);
    float* accum = accum_tensor_.data<float>();
    const size_t grid_dim = (len - 1) / block_dim + 1;
    ada_grad_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, wgrad, accum, lr_, epsilon_, scaler_);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class AdaGradOptimizer<float>;
template class AdaGradOptimizer<__half>;
}  // namespace HugeCTR
