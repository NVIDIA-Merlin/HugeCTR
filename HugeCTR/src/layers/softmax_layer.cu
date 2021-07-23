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

  batchsize = in_tensor.get_dimensions()[0];
  len = in_tensors_[0].get_num_elements();
  hiddensize = len / batchsize;
  blobs_buff->reserve({batchsize}, &workspace);
}

template <typename T>
void __global__ Softmax_fprop_kernel(T* out, T* workspace, int m, int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    out[offset + tid] = out[offset + tid] / workspace[blockIdx.x];
  }
}

template <typename T>
void Softmax_fprop(T* out, T* workspace, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Softmax_fprop_kernel<<<grid, block, 0, stream>>>(out, workspace, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void __global__ Softmax_bprop_kernel(T* top, T* bottom, T* workspace, int m, int n) {
  int offset = blockIdx.x * n;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int idx = offset + tid;
    bottom[idx] = bottom[idx] / workspace[blockIdx.x] *
                  (1 - bottom[idx] / pow(workspace[blockIdx.x], 2.0)) * top[idx];
  }
}

template <typename T>
void Softmax_bprop(T* top, T* bottom, T* workspace, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  Softmax_bprop_kernel<<<grid, block, 0, stream>>>(top, bottom, workspace, m, n);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void SoftmaxLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];
  const auto& in_tensor_dim = in_tensor.get_dimensions();
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(out_tensor.get_ptr(), in_tensor.get_ptr(), len,
                            [] __device__(T in) { return expf(in); }, get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1].
  MLCommon::LinAlg::reduce(workspace.get_ptr(), out_tensor.get_ptr(), hiddensize, batchsize, T(0),
                           true, true, get_gpu().get_stream());
  // Softmax exp(x_i) / sum(exp)(x_i)) i=[0, embedding_vector_size-1].
  Softmax_fprop(out_tensor.get_ptr(), workspace.get_ptr(), in_tensor_dim[0], in_tensor_dim[1],
                get_gpu().get_stream());
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
void SoftmaxLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& bottom_tensor = in_tensors_[0];
  Tensor2<T>& top_tensor = out_tensors_[0];
  const auto& in_tensor_dim = bottom_tensor.get_dimensions();
  // exp(x_i)
  MLCommon::LinAlg::unaryOp(bottom_tensor.get_ptr(), bottom_tensor.get_ptr(), len,
                            [] __device__(T in) { return expf(in); }, get_gpu().get_stream());
  // Get sum of exp(x_i) i=[0, embedding_vector_size-1].
  MLCommon::LinAlg::reduce(workspace.get_ptr(), bottom_tensor.get_ptr(), hiddensize, batchsize,
                           T(0), true, true, get_gpu().get_stream());
  // softmax derivative :
  // q(x) = e^xi / sum(e^xi);
  // f(x) = q(x) * (1 - q(x) / sum(e^xi));
  // Y = f(x)*d_top i = [0, embedding_vector_size - 1];
  Softmax_bprop(top_tensor.get_ptr(), bottom_tensor.get_ptr(), workspace.get_ptr(),
                in_tensor_dim[0], in_tensor_dim[1], get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class SoftmaxLayer<float>;

}  // namespace HugeCTR
