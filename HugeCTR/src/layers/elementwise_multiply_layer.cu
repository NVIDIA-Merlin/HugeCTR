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
    const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "ElementwiseMultiplyLayer needs at least 2 input tensors");
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
    blobs_buff->reserve(out_tensor.get_dimensions(), &fprop_output_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ElementwiseMultiplyLayer<T>::initialize() {
  std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> pinned_host_buf =
      GeneralBuffer2<CudaHostAllocator>::create();
  pinned_host_buf->reserve({num_}, &h_inputs_);
  pinned_host_buf->allocate();
}

template <typename T>
void ElementwiseMultiplyLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  T* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      d_inputs_.get_ptr(), output, size_, num_);

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_output_.get_ptr(), (void*)output,
                                 out_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <>
void ElementwiseMultiplyLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(__half*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  __half* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ / 2 + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      d_inputs_.get_ptr(), output, size_, num_);

  HCTR_LIB_THROW(cudaMemcpyAsync((void*)fprop_output_.get_ptr(), (void*)output,
                                 out_tensors_[0].get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void ElementwiseMultiplyLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(T*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  T* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      output, d_inputs_.get_ptr(), fprop_output_.get_ptr(), size_, num_);
}

template <>
void ElementwiseMultiplyLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    HCTR_LIB_THROW(cudaMemcpyAsync((void*)d_inputs_.get_ptr(), (void*)h_inputs_.get_ptr(),
                                   num_ * sizeof(__half*), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
    initialized_ = true;
  }
  __half* output = out_tensors_[0].get_ptr();

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size_ / 2 + blockSize.x - 1) / blockSize.x, 1, 1);
  elementwise_multiply_dgrad_kernel<<<gridSize, blockSize, 0, get_gpu().get_stream()>>>(
      output, d_inputs_.get_ptr(), fprop_output_.get_ptr(), size_, num_);
}

template class ElementwiseMultiplyLayer<float>;
template class ElementwiseMultiplyLayer<__half>;

}  // namespace HugeCTR
