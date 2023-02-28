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
#include <optimizers/nesterov_optimizer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void nesterov_update_kernel(int len, float* weight, float* accum, const T* wgrad,
                                       float lr, float mu, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float accum_old = accum[i];
    float accum_new = mu * accum_old - lr * TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    accum[i] = accum_new;
    weight[i] += (-mu * accum_old + (1.f + mu) * accum_new);
  }
}

}  // namespace

template <typename T>
NesterovOptimizer<T>::NesterovOptimizer(const Tensor2<float>& weight_main, const Tensor2<T>& wgrad,
                                        const std::shared_ptr<BufferBlock2<float>>& opt_buf,
                                        const std::shared_ptr<GPUResource>& gpu_resource,
                                        float learning_rate, float momentum_factor, float scaler)
    : Optimizer(weight_main, gpu_resource, learning_rate, scaler),
      wgrad_(wgrad),
      wgrad_tensors_({}),
      mu_(momentum_factor) {
  if (weight_main_.get_num_elements() != wgrad_.get_num_elements()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "weight->get_num_elements() != wgrad->get_num_elements()");
  }
  opt_buf->reserve({weight_main.get_num_elements()}, &accum_);
}
template <typename T>
NesterovOptimizer<T>::NesterovOptimizer(std::vector<core23::Tensor> weight_tensors,
                                        std::vector<core23::Tensor> wgrad_tensors,
                                        const std::shared_ptr<GPUResource>& gpu_resource,
                                        float learning_rate, float momentum_factor, float scaler)
    : Optimizer(weight_tensors, gpu_resource, learning_rate, scaler),
      wgrad_tensors_(std::make_optional<WgradTensors<T>>(
          std::move(wgrad_tensors), core23::Shape({static_cast<int64_t>(wgrad_tensors.size())}))),
      mu_(momentum_factor) {
  core23::TensorParams tensor_params =
      core23::TensorParams()
          .device(core23::Device(core23::DeviceType::GPU, gpu_resource->get_device_id()))
          .data_type(core23::ScalarType::Float)
          .shape(core23::Shape({weight_tensors_->flatten().size(0)}))
          .buffer_channel(GetOptStateBufferChannnel());

  accum_tensor_ = core23::Tensor(tensor_params);
}

template <typename T>
void NesterovOptimizer<T>::initialize() {
  if (!wgrad_tensors_) {
    HCTR_LIB_THROW(cudaMemsetAsync(accum_.get_ptr(), 0, accum_.get_size_in_bytes(),
                                   gpu_resource_->get_stream()));
  } else {
    HCTR_LIB_THROW(cudaMemsetAsync(accum_tensor_.data(), 0, accum_tensor_.num_bytes(),
                                   gpu_resource_->get_stream()));
  }
}

template <typename T>
void NesterovOptimizer<T>::update() {
  CudaDeviceContext context(get_device_id());

  constexpr size_t block_dim = 256;

  if (!wgrad_tensors_) {
    const size_t len = weight_main_.get_num_elements();
    const size_t grid_dim = (len - 1) / block_dim + 1;
    float* weight = weight_main_.get_ptr();
    float* accum = accum_.get_ptr();
    T* wgrad = wgrad_.get_ptr();
    nesterov_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, accum, wgrad, lr_, mu_, scaler_);
  } else {
    auto flat_weight_tensor = weight_tensors_->flatten();
    auto flat_wgrad_tensor = wgrad_tensors_->flatten();
    float* weight = flat_weight_tensor.data();
    const T* wgrad = flat_wgrad_tensor.data();
    auto len = flat_weight_tensor.size(0);
    float* accum = accum_tensor_.data<float>();
    const size_t grid_dim = (len - 1) / block_dim + 1;
    nesterov_update_kernel<<<grid_dim, block_dim, 0, gpu_resource_->get_stream()>>>(
        len, weight, accum, wgrad, lr_, mu_, scaler_);
  }

#ifndef NDEBUG
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class NesterovOptimizer<float>;
template class NesterovOptimizer<__half>;

}  // namespace HugeCTR
