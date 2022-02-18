/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <layers/relu_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

__global__ void forward_half2_relu_kernel(__half* top, const __half* bottom, int size) {
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* top2 = reinterpret_cast<__half2*>(top);
  const __half2* bottom2 = reinterpret_cast<const __half2*>(bottom);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    __half2 t = __ldg(bottom2 + i);
    __half2 mask = __hgt2(t, zero);
    top2[i] = __hmul2(t, mask);
  }
}

__global__ void backward_half2_relu_kernel(__half* bottom, const __half* top, int size) {
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* bottom2 = reinterpret_cast<__half2*>(bottom);
  const __half2* top2 = reinterpret_cast<const __half2*>(top);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    __half2 t = bottom2[i];
    half2 mask = __hgt2(t, zero);
    bottom2[i] = __hmul2(__ldg(top2 + i), mask);
  }
}

}  // namespace

template <typename T>
ReluLayer<T>::ReluLayer(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                        const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(in_tensor.get_num_elements() % 2 == 0);

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void ReluLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  int len = in_tensors_[0].get_num_elements();

  auto fop = [] __device__(T in) { return (in > T(0)) ? in : T(0); };

  MLCommon::LinAlg::unaryOp(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), len, fop,
                            get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void ReluLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  int len = in_tensors_[0].get_num_elements();

  auto bop = [] __device__(T d_out, T d_in) { return (d_in > T(0)) ? d_out : T(0); };

  MLCommon::LinAlg::binaryOp(in_tensors_[0].get_ptr(), out_tensors_[0].get_ptr(),
                             in_tensors_[0].get_ptr(), len, bop, get_gpu().get_stream());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

ReluLayer<__half>::ReluLayer(const Tensor2<__half>& bottom_tensor,
                             const Tensor2<__half>& top_tensor,
                             const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(get_size_from_dims(bottom_tensor.get_dimensions()) ==
         get_size_from_dims(top_tensor.get_dimensions()));
  assert(get_size_from_dims(bottom_tensor.get_dimensions()) % 2 == 0);

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void ReluLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = bottom_tensor_.get_num_elements() / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  forward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
      top_tensor_.get_ptr(), bottom_tensor_.get_ptr(), size);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

void ReluLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = bottom_tensor_.get_num_elements() / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  backward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
      bottom_tensor_.get_ptr(), top_tensor_.get_ptr(), size);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class ReluLayer<float>;
template class ReluLayer<__half>;

}  // namespace HugeCTR
