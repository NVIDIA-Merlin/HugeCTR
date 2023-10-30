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

#include <cuda_fp16.h>

#include <algorithm>
#include <functional>
#include <layers/elementwise_multiply_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

#define BLOCK_DIM_SIZE 32

template <typename T>
__global__ void elementwise_multiply_kernel(T** inputs, T* output, int size, int num) {
  T one = 1.0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    T tmp = one;
    for (int i = 0; i < num; i++) {
      tmp *= inputs[i][tid];
    }
    output[tid] = tmp;
  }
}

template <>
__global__ void elementwise_multiply_kernel<__half>(__half** inputs, __half* output, int size,
                                                    int num) {
  __half2** inputs2 = reinterpret_cast<__half2**>(inputs);
  __half2* output2 = reinterpret_cast<__half2*>(output);
  const int size2 = size / 2;

  const __half one = __float2half(1.0f);
  const __half2 one2 = __half2half2(one);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size2) {
    __half2 tmp2 = one2;
    for (int i = 0; i < num; i++) {
      tmp2 *= inputs2[i][tid];
    }
    output2[tid] = tmp2;
  }
  if (tid == 0 && size % 2 > 0) {
    __half tmp = one;
    for (int i = 0; i < num; i++) {
      tmp *= inputs[i][size - 1];
    }
    output[size - 1] = tmp;
  }
}

template <typename T>
__global__ void elementwise_multiply_dgrad_kernel(const T* top_grad, T** dgrads, T* fprop_output,
                                                  int size, int num) {
  T zero = 0.0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    for (int i = 0; i < num; ++i) {
      if (fprop_output[tid] == zero) {
        dgrads[i][tid] = zero;
      } else {
        T d_input = dgrads[i][tid];
        dgrads[i][tid] = top_grad[tid] * (fprop_output[tid] / d_input);
      }
    }
  }
}

template <>
__global__ void elementwise_multiply_dgrad_kernel<__half>(const __half* top_grad, __half** dgrads,
                                                          __half* fprop_output, int size, int num) {
  const __half2* top_grad2 = reinterpret_cast<const __half2*>(top_grad);
  __half2** dgrads2 = reinterpret_cast<__half2**>(dgrads);
  __half2* fprop_output2 = reinterpret_cast<__half2*>(fprop_output);
  const int size2 = size / 2;

  const __half zero = __float2half(0.0f);
  const __half2 zero2 = __half2half2(zero);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size2) {
    for (int i = 0; i < num; ++i) {
      const __half2 fp_out2 = fprop_output2[tid];
      const __half2 pred2 = __hne2(fp_out2, zero2);
      dgrads2[i][tid] = pred2 * top_grad2[tid] * (fp_out2 / dgrads2[i][tid]);
    }
  }
  if (tid == 0 && size % 2 > 0) {
    const int idx = size - 1;
    for (int i = 0; i < num; ++i) {
      const __half fp_out = fprop_output[idx];
      dgrads[i][idx] = (fp_out == zero) ? zero : top_grad[idx] * (fp_out / dgrads[i][idx]);
    }
  }
}

}  // end of namespace

template <typename T>
ElementwiseMultiplyLayer<T>::ElementwiseMultiplyLayer(
    const std::vector<core23::Tensor>& in_tensors, const core23::Tensor& out_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    size_ = in_tensors[0].num_elements();
    num_ = in_tensors.size();
    core23::BufferParams blobs_buffer_params = {};
    blobs_buffer_params.channel = GetBlobsBufferChannel();
    // error input checking
    auto dims = in_tensors[0].shape();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "ElementwiseMultiplyLayer needs at least 2 input tensors");
    }
    for (int64_t i = 1; i < num_; i++) {
      if (in_tensors[i].shape().dims() != dims.dims()) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.dims(); j++) {
        if (in_tensors[i].shape().size(j) != dims[j]) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (int64_t i = 0; i < num_; i++) {
      input_tensors_.push_back(in_tensors[i]);
    }
    output_tensors_.push_back(out_tensor);
    d_inputs_ =
        core23::Tensor(core23::TensorParams()
                           .buffer_params(blobs_buffer_params)
                           .data_type(core23::ToScalarType<void*>::value)
                           .shape({num_})
                           .device(core23::Device(core23::DeviceType::GPU, this->get_device_id())));
    fprop_output_ =
        core23::Tensor(core23::TensorParams()
                           .buffer_params(blobs_buffer_params)
                           .data_type(core23::ToScalarType<T>::value)
                           .shape(out_tensor.shape())
                           .device(core23::Device(core23::DeviceType::GPU, this->get_device_id())));

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ElementwiseMultiplyLayer<T>::initialize() {
  h_inputs_ = core23::Tensor(core23::TensorParams()
                                 .data_type(core23::ToScalarType<void*>::value)
                                 .shape({num_})
                                 .device(core23::Device(core23::DeviceType::CPU)));
  h_inputs_.data();

  for (int64_t i = 0; i < num_; i++) {
    // data address
    uint64_t* to_write = h_inputs_.data<uint64_t>() + i;
    *to_write = reinterpret_cast<uint64_t>(input_tensors_[i].data());
  }
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.data(), (void*)h_inputs_.data(),
                                 num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));
  initialized_ = true;
}

template <typename T>
void ElementwiseMultiplyLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  T* output = output_tensors_[0].data<T>();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      d_inputs_.data<T*>(), output, size_, num_);

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_output_.data(), (void*)output,
                                 output_tensors_[0].num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <>
void ElementwiseMultiplyLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  __half* output = output_tensors_[0].data<__half>();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ / 2 + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      d_inputs_.data<__half*>(), output, size_, num_);

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_output_.data(), (void*)output,
                                 output_tensors_[0].num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void ElementwiseMultiplyLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  T* output = output_tensors_[0].data<T>();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      output, d_inputs_.data<T*>(), fprop_output_.data<T>(), size_, num_);
}

template <>
void ElementwiseMultiplyLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());
  __half* output = output_tensors_[0].data<__half>();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ / 2 + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      output, d_inputs_.data<__half*>(), fprop_output_.data<__half>(), size_, num_);
}

template class ElementwiseMultiplyLayer<float>;
template class ElementwiseMultiplyLayer<__half>;

}  // namespace HugeCTR
