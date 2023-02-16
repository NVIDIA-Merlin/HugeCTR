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
#include <functional>
#include <layers/add_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename ContainerView, typename T>
__global__ void add_kernel(ContainerView inputs, T* output, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    T tmp = 0;
    for (int i = 0; i < inputs.size(0); i++) {
      tmp += inputs[i][tid];
    }
    output[tid] = tmp;
  }
}

template <typename T>
__global__ void add_kernel(T** inputs, T* output, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    T tmp = 0;
    for (int i = 0; i < num; i++) {
      tmp += inputs[i][tid];
    }
    output[tid] = tmp;
  }
}

template <typename ContainerView, typename T>
__global__ void add_dgrad_kernel(const T* top_grad, ContainerView dgrads, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < dgrads.size(0); i++) {
      dgrads[i][tid] = top_grad[tid];
    }
  }
}

template <typename T>
__global__ void add_dgrad_kernel(const T* top_grad, T** dgrads, int size, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < num; i++) {
      dgrads[i][tid] = top_grad[tid];
    }
  }
}

template <>
__global__ void add_kernel<core23::TensorContainer<__half, 1, 1>::View, __half>(
    core23::TensorContainer<__half, 1, 1>::View inputs, __half* output, int size) {
  __half2* output2 = (__half2*)(output);
  int size2 = size / 2;

  const __half2 zero = __half2half2(__float2half(0.f));
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = start; i < size2; i += stride) {
    __half2 tmp = zero;
    for (int j = 0; j < inputs.size(0); ++j) {
      const __half2* input2 = reinterpret_cast<__half2*>(&inputs[j][0]);
      tmp += input2[i];
    }
    output2[i] = tmp;
  }
  if (start == 0 && size % 2 > 0) {
    __half tmp = __float2half(0.0f);
    for (int j = 0; j < inputs.size(0); ++j) {
      tmp += inputs[j][size - 1];
    }
    output[size - 1] = tmp;
  }
}

template <>
__global__ void add_kernel<__half>(__half** inputs, __half* output, int size, int num) {
  const __half2** inputs2 = (const __half2**)(inputs);
  __half2* output2 = (__half2*)(output);
  int size2 = size / 2;

  const __half2 zero = __half2half2(__float2half(0.f));
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = start; i < size2; i += stride) {
    __half2 tmp = zero;
    for (int j = 0; j < num; ++j) {
      tmp += inputs2[j][i];
    }
    output2[i] = tmp;
  }
  if (start == 0 && size % 2 > 0) {
    __half tmp = __float2half(0.0f);
    for (int j = 0; j < num; ++j) {
      tmp += inputs[j][size - 1];
    }
    output[size - 1] = tmp;
  }
}

template <>
__global__ void add_dgrad_kernel<core23::TensorContainer<__half, 1, 1>::View, __half>(
    const __half* top_grad, core23::TensorContainer<__half, 1, 1>::View dgrads, int size) {
  const __half2* top_grad2 = (const __half2*)(top_grad);
  int size2 = size / 2;

  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = start; i < size2; i += stride) {
    for (int j = 0; j < dgrads.size(0); ++j) {
      __half2* dgrad2 = reinterpret_cast<__half2*>(&dgrads[j][0]);
      dgrad2[i] = top_grad2[i];
    }
  }
  if (start == 0 && size % 2 > 0) {
    for (int j = 0; j < dgrads.size(0); ++j) {
      dgrads[j][size - 1] = top_grad[size - 1];
    }
  }
}

template <>
__global__ void add_dgrad_kernel<__half>(const __half* top_grad, __half** dgrads, int size,
                                         int num) {
  const __half2* top_grad2 = (const __half2*)(top_grad);
  __half2** dgrads2 = (__half2**)(dgrads);
  int size2 = size / 2;

  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = start; i < size2; i += stride) {
    for (int j = 0; j < num; ++j) {
      dgrads2[j][i] = top_grad2[i];
    }
  }
  if (start == 0 && size % 2 > 0) {
    for (int j = 0; j < num; ++j) {
      dgrads[j][size - 1] = top_grad[size - 1];
    }
  }
}

}  // end of namespace

template <typename T>
AddLayer<T>::AddLayer(const std::vector<core23::Tensor>& input_tensors,
                      const core23::Tensor& output_tensor,
                      const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(input_tensors, {output_tensor}, gpu_resource),
      size_(input_tensors_[0].num_elements()),
      input_tensor_container_(input_tensors_, {static_cast<int64_t>(input_tensors_.size())}) {
  try {
    // error input checking
    int64_t dims = input_tensors_[0].dims();
    int64_t num = input_tensors_.size();
    if (num < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "AddLayer needs at least 2 input tensors");
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
AddLayer<T>::AddLayer(const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
                      const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                      const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "AddLayer needs at least 2 input tensors");
    }
    for (size_t i = 1; i < num_; i++) {
      if (in_tensors[i].get_dimensions().size() != dims.size()) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.size(); j++) {
        if (in_tensors[i].get_dimensions()[j] != dims[j]) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve({num_}, &d_inputs_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void AddLayer<T>::initialize() {
  // TODO: this block will be removed later
  if (input_tensors_.empty()) {
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> pinned_host_buf =
        GeneralBuffer2<CudaHostAllocator>::create();
    pinned_host_buf->reserve({num_}, &h_inputs_);
    pinned_host_buf->allocate();

    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }

    HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
  } else {
    input_tensor_container_.flatten();
  }
}

template <typename T>
void AddLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  // TODO: this block will be removed later
  if (input_tensors_.empty()) {
    T* output = out_tensors_[0].get_ptr();

    dim3 block_size(256, 1, 1);
    dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
    add_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(d_inputs_.get_ptr(), output,
                                                                     size_, num_);
  } else {
    T* output = output_tensors_[0].data<T>();

    dim3 block_size(256, 1, 1);
    dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
    add_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(input_tensor_container_.view(),
                                                                     output, size_);
  }
}

template <typename T>
void AddLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  // TODO: this block will be removed later
  if (input_tensors_.empty()) {
    T* output = out_tensors_[0].get_ptr();

    dim3 block_size(256, 1, 1);
    dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
    add_dgrad_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        output, d_inputs_.get_ptr(), size_, num_);
  } else {
    T* output = output_tensors_[0].data<T>();

    dim3 block_size(256, 1, 1);
    dim3 grid_size((size_ + block_size.x - 1) / block_size.x, 1, 1);
    add_dgrad_kernel<<<grid_size, block_size, 0, get_gpu().get_stream()>>>(
        output, input_tensor_container_.view(), size_);
  }
}

template class AddLayer<float>;
template class AddLayer<__half>;

}  // namespace HugeCTR
