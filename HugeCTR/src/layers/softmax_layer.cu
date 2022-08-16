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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/softmax_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/reduce.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
SoftmaxLayer<T>::SoftmaxLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);

  len_ = in_tensors_[0].get_num_elements();
  dims_ = in_tensor.get_dimensions().size();
  hidden_size_ = in_tensor.get_dimensions()[dims_ - 1];
  n_rows_ = len_ / hidden_size_;
  blobs_buff->reserve({n_rows_}, &workspace_);
  blobs_buff->reserve({hidden_size_}, &identity_);
  blobs_buff->reserve(in_tensor.get_dimensions(), &softmax_out_);
}

template <typename T>
void SoftmaxLayer<T>::initialize() {
  CudaDeviceContext context(get_device_id());
  initialize_array<<<(hidden_size_ - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      identity_.get_ptr(), hidden_size_, 1.0f);
}

template <>
void SoftmaxLayer<__half>::initialize() {
  CudaDeviceContext context(get_device_id());
  initialize_array<<<(hidden_size_ - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      identity_.get_ptr(), hidden_size_, __float2half(1.0f));
}

template <typename T>
void __global__ Softmax_fprop_kernel(T* out, T* workspace, int m, int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    out[offset + tid] = out[offset + tid] / workspace[blockIdx.x];
  }
}

template <>
void __global__ Softmax_fprop_kernel(__half* out, __half* workspace, int m, int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    out[offset + tid] = __hdiv(out[offset + tid], workspace[blockIdx.x]);
  }
}

template <typename T>
void Softmax_fprop(T* out, T* workspace, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Softmax_fprop_kernel<<<grid, block, 0, stream>>>(out, workspace, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void __global__ Softmax_bprop_kernel(T* top, T* bottom, T* softmax, int m, int n) {
  int offset = blockIdx.x * n;
  float grad_softmax = static_cast<float>(0.0f);
  __shared__ float grad_sum;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    grad_softmax += top[idx] * softmax[idx];
  }
  float tmp = blockReduceSum<T>(grad_softmax);
  if (threadIdx.x == 0) {
    grad_sum = tmp;
  }
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    bottom[idx] = softmax[idx] * top[idx] - softmax[idx] * grad_sum;
  }
}

template <>
void __global__ Softmax_bprop_kernel(__half* top, __half* bottom, __half* softmax, int m, int n) {
  int offset = blockIdx.x * n;
  float grad_softmax = static_cast<float>(0.0f);
  __shared__ __half grad_sum;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    grad_softmax += static_cast<float>(top[idx] * softmax[idx]);
  }

  float tmp = blockReduceSum<float>(grad_softmax);
  if (threadIdx.x == 0) {
    grad_sum = static_cast<__half>(tmp);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    __half tmp = __hsub(top[idx], grad_sum);
    bottom[idx] = __hmul(bottom[idx], tmp);
  }
}

template <typename T>
void Softmax_bprop(T* top, T* bottom, T* softmax_out, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Softmax_bprop_kernel<<<grid, block, 0, stream>>>(top, bottom, softmax_out, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void SoftmaxLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(
      out_tensor.get_ptr(), in_tensor.get_ptr(), len_, [] __device__(T in) { return expf(in); },
      get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1].
  MLCommon::LinAlg::reduce(workspace_.get_ptr(), out_tensor.get_ptr(), hidden_size_, n_rows_, T(0),
                           true, true, get_gpu().get_stream());
  // Softmax exp(x_i) / sum(exp)(x_i)) i=[0, embedding_vector_size-1].
  Softmax_fprop(out_tensor.get_ptr(), workspace_.get_ptr(), n_rows_, hidden_size_,
                get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)softmax_out_.get_ptr(), (void*)out_tensor.get_ptr(),
                                 out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <>
void SoftmaxLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<__half>& in_tensor = in_tensors_[0];
  Tensor2<__half>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();

  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(
      out_tensor.get_ptr(), in_tensor.get_ptr(), len_,
      [] __device__(__half in) { return hexp(in); }, get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1]
  HCTR_LIB_THROW(cublasGemmEx(
      get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n_rows_, 1, hidden_size_, &alpha,
      out_tensor.get_ptr(), CUDA_R_16F, hidden_size_, identity_.get_ptr(), CUDA_R_16F, hidden_size_,
      &beta, workspace_.get_ptr(), CUDA_R_16F, n_rows_, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  // Softmax exp(x_i) / sum(exp)(x_i)) i=[0, embedding_vector_size-1]
  Softmax_fprop(out_tensor.get_ptr(), workspace_.get_ptr(), n_rows_, hidden_size_,
                get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)softmax_out_.get_ptr(), (void*)out_tensor.get_ptr(),
                                 out_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void SoftmaxLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& bottom_tensor = in_tensors_[0];
  Tensor2<T>& top_tensor = out_tensors_[0];
  const auto& in_tensor_dim = bottom_tensor.get_dimensions();

  const size_t len = bottom_tensor.get_num_elements();

  Softmax_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), softmax_out_.get_ptr(), n_rows_,
                hidden_size_, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <>
void SoftmaxLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<__half>& bottom_tensor = in_tensors_[0];
  Tensor2<__half>& top_tensor = out_tensors_[0];
  const auto& in_tensor_dim = bottom_tensor.get_dimensions();

  Softmax_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), softmax_out_.get_ptr(), n_rows_,
                hidden_size_, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<__half>;

}  // namespace HugeCTR
