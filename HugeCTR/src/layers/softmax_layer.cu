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
#include <include/utils.cuh>
#include <layers/softmax_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/reduce.cuh>
#include <linalg/unary_op.cuh>
#include <network_buffer_channels.hpp>
#include <utils.hpp>
namespace HugeCTR {

template <typename T>
SoftmaxLayer<T>::SoftmaxLayer(const core23::Tensor& input_tensor,
                              const core23::Tensor& output_tensor,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource) {
  assert(input_tensor.num_elements() == output_tensor.num_elements());

  len_ = input_tensors_[0].num_elements();
  dims_ = input_tensor.shape().dims();
  hidden_size_ = input_tensor.shape().size(dims_ - 1);
  n_rows_ = len_ / hidden_size_;
  core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};
  auto param = (input_tensor.my_params().buffer_params(buf_p));
  workspace23_ = core23::Tensor(
      param.shape({(int64_t)n_rows_}).data_type(core23::DataType(core23::ToScalarType<T>::value)));
  identity23_ = core23::Tensor(param.shape({(int64_t)hidden_size_})
                                   .data_type(core23::DataType(core23::ToScalarType<T>::value)));
  softmax_out23_ = core23::Tensor(param.shape(input_tensor.shape())
                                      .data_type(core23::DataType(core23::ToScalarType<T>::value)));
}
template <typename T>
void SoftmaxLayer<T>::initialize() {
  CudaDeviceContext context(get_device_id());

  initialize_array<<<(hidden_size_ - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      identity23_.data<T>(), hidden_size_, 1.0f);
}

template <>
void SoftmaxLayer<__half>::initialize() {
  CudaDeviceContext context(get_device_id());

  initialize_array<<<(hidden_size_ - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(
      identity23_.data<__half>(), hidden_size_, __float2half(1.0f));
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
  float grad_softmax = 0.f;
  __shared__ float grad_sum;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    grad_softmax += __half2float(top[idx]) * __half2float(softmax[idx]);
  }

  float tmp = blockReduceSum<float>(grad_softmax);
  if (threadIdx.x == 0) {
    grad_sum = tmp;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    bottom[idx] =
        __half2float(softmax[idx]) * __half2float(top[idx]) - __half2float(softmax[idx]) * grad_sum;
  }
}

template <typename T>
void Softmax_bprop(T* top, T* bottom, T* softmax_out, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Softmax_bprop_kernel<<<grid, block, 0, stream>>>(top, bottom, softmax_out, m, n);
}

template <typename T>
void SoftmaxLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  core23::Tensor& input_tensor = input_tensors_[0];
  core23::Tensor& output_tensor = output_tensors_[0];
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(
      output_tensor.data<T>(), input_tensor.data<T>(), len_,
      [] __device__(T in) { return expf(in); }, get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1].
  MLCommon::LinAlg::reduce(workspace23_.data<T>(), output_tensor.data<T>(), hidden_size_, n_rows_,
                           T(0), true, true, get_gpu().get_stream());
  // Softmax exp(x_i) / sum(exp)(x_i)) i=[0, embedding_vector_size-1].
  Softmax_fprop(output_tensor.data<T>(), workspace23_.data<T>(), n_rows_, hidden_size_,
                get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync(softmax_out23_.data(), output_tensor.data(),
                                 output_tensor.num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <>
void SoftmaxLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  core23::Tensor& input_tensor = input_tensors_[0];
  core23::Tensor& output_tensor = output_tensors_[0];

  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(
      output_tensor.data<__half>(), input_tensor.data<__half>(), len_,
      [] __device__(__half in) { return hexp(in); }, get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1]
  HCTR_LIB_THROW(cublasGemmEx(
      get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n_rows_, 1, hidden_size_, &alpha,
      output_tensor.data(), CUDA_R_16F, hidden_size_, identity23_.data(), CUDA_R_16F, hidden_size_,
      &beta, workspace23_.data(), CUDA_R_16F, n_rows_, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  // Softmax exp(x_i) / sum(exp)(x_i)) i=[0, embedding_vector_size-1]
  Softmax_fprop(output_tensor.data<__half>(), workspace23_.data<__half>(), n_rows_, hidden_size_,
                get_gpu().get_stream());
  HCTR_LIB_THROW(cudaMemcpyAsync(softmax_out23_.data(), output_tensor.data(),
                                 output_tensor.num_bytes(), cudaMemcpyDeviceToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void SoftmaxLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  core23::Tensor& bottom_tensor = input_tensors_[0];
  core23::Tensor& top_tensor = output_tensors_[0];

  Softmax_bprop(top_tensor.data<T>(), bottom_tensor.data<T>(), softmax_out23_.data<T>(), n_rows_,
                hidden_size_, get_gpu().get_stream());
}

template <>
void SoftmaxLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  core23::Tensor& bottom_tensor = input_tensors_[0];
  core23::Tensor& top_tensor = output_tensors_[0];

  Softmax_bprop(top_tensor.data<__half>(), bottom_tensor.data<__half>(),
                softmax_out23_.data<__half>(), n_rows_, hidden_size_, get_gpu().get_stream());
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<__half>;

}  // namespace HugeCTR
