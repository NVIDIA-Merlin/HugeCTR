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

#include <algorithm>
#include <core23/tensor_operations.hpp>
#include <functional>
#include <layers/sub_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>
namespace HugeCTR {

namespace {

template <typename T>
__global__ void sub_kernel(T** inputs, T* output, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) output[tid] = inputs[0][tid] - inputs[1][tid];
}

template <typename T>
__global__ void sub_dgrad_kernel(const T* top_grad, T** dgrads, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    dgrads[0][tid] = top_grad[tid];
    dgrads[1][tid] = 0.0 - top_grad[tid];
  }
}

}  // end of namespace

template <typename T>
SubLayer<T>::SubLayer(const std::vector<core23::Tensor>& input_tensors,
                      const core23::Tensor& output_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(input_tensors, {output_tensor}, gpu_resource), size_(input_tensors_[0].num_elements()) {
  try {
    // error input checking
    int64_t dims = input_tensors_[0].dims();
    int64_t num = input_tensors_.size();
    if (num != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "SubLayer needs 2 input tensors");
    }
    for (auto i = 1; i < num; i++) {
      if (input_tensors_[i].dims() != dims) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (auto j = 0; j < dims; j++) {
        if (input_tensors_[i].size(j) != input_tensors_[0].size(j)) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void SubLayer<T>::initialize() {
  CudaDeviceContext context(get_device_id());

  core23::TensorParams ptr_params =
      core23::TensorParams()
          .shape({static_cast<int64_t>(input_tensors_.size())})
          .data_type(core23::ScalarType::Pointer)
          .device({core23::DeviceType::GPU, static_cast<int8_t>(this->get_device_id())});
  input_tensor_ptr_ = core23::Tensor(ptr_params);
  std::vector<void*> ptr_cpu;
  // the in_tensors_ must be allocated before initialize() is called
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    ptr_cpu.push_back(input_tensors_[i].data());
  }
  core23::copy_async(input_tensor_ptr_, ptr_cpu, get_gpu().get_stream());
}

template <typename T>
void SubLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  T* output = output_tensors_[0].data<T>();

  dim3 block_size(256, 1, 1);
  dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
  sub_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(input_tensor_ptr_.data<T*>(),
                                                                   output, size_);
}

template <typename T>
void SubLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  T* output = output_tensors_[0].data<T>();
  dim3 block_size(256, 1, 1);
  dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
  sub_dgrad_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
      output, input_tensor_ptr_.data<T*>(), size_);
}

template class SubLayer<float>;

}  // namespace HugeCTR
