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

#include "HugeCTR/include/layers/relu_layer.hpp"

#include "HugeCTR/include/layers/element_wise_function.hpp"
#include "HugeCTR/include/utils.cuh"

#include <algorithm>
#include <functional>
#include "HugeCTR/include/utils.hpp"
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
ReluLayer<T>::ReluLayer(const std::shared_ptr<Tensor<T>>& in_tensor,
                        const std::shared_ptr<Tensor<T>>& out_tensor, int device_id)
    : Layer(device_id) {
  assert(get_size_from_dims(in_tensor->get_dims()) == get_size_from_dims(out_tensor->get_dims()));
  assert(get_size_from_dims(in_tensor->get_dims()) % 2 == 0);

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);
}

template <>
void ReluLayer<float>::fprop(cudaStream_t stream) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  auto fop = [] __device__(float in) { return (in < 0) ? 0 : in; };
  internal::ElementWiseFunctor functor;
  functor.forward_evaluate(*in_tensor, *out_tensor, get_device_id(), fop, stream);
}

template <>
void ReluLayer<__half>::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = get_size_from_dims(in_tensors_[0]->get_dims()) / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  forward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(out_tensors_[0]->get_ptr(),
                                                                in_tensors_[0]->get_ptr(), size);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <>
void ReluLayer<float>::bprop(cudaStream_t stream) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  auto bop = [] __device__(float d_out, float d_in) { return (d_in < 0) ? 0 : d_out; };
  internal::ElementWiseFunctor functor;
  functor.backward_evaluate(*in_tensor, *out_tensor, get_device_id(), bop, stream);
}

template <>
void ReluLayer<__half>::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = get_size_from_dims(in_tensors_[0]->get_dims()) / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  backward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(in_tensors_[0]->get_ptr(),
                                                                 out_tensors_[0]->get_ptr(), size);
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class ReluLayer<float>;
template class ReluLayer<__half>;

}  // namespace HugeCTR
