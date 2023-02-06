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
#pragma once

#include <algorithm>
#include <functional>
#include <tensor2.hpp>
#include <utils.hpp>

namespace HugeCTR {
namespace internal {

const int BLOCK_SIZE = 512;
const int MAX_GRID_SIZE = 1024;

template <typename Fop>
__global__ void forward_element_wise_kernel(const float* in, float* out, int len, Fop fop) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    out[i] = fop(in[i]);
  }
}

template <typename Bop>
__global__ void backward_element_wise_kernel(const float* d_out, float* d_in, int len, Bop bop) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    d_in[i] = bop(d_out[i], d_in[i]);
  }
}

/**
 * Common implementation for the element wise layers such as Relu and Elu.
 * Their fprop/brop are just the wrapperw of forward_evaluate/backward_evaluate,
 * while passing the simple scalar lambda operations to them.
 * All the other element wise layers can be implementated in the similar way.
 */
class ElementWiseFunctor {
 public:
  /**
   * Ctor of ElementWiseFunctor. Copy construction and assigment are disabled.
   */
  ElementWiseFunctor() {}
  ElementWiseFunctor(const ElementWiseFunctor&) = delete;
  ElementWiseFunctor& operator=(const ElementWiseFunctor&) = delete;

  /**
   * D'tor of ElementWiseFunctor.
   */
  ~ElementWiseFunctor() {}

  /**
   * A method of implementing the element-wise forward pass
   * @tparam Fop the type of simple scalar lambda operation
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this operation is handled
   * @param fop Fop lambda object to do the operation per element
   * @param stream CUDA stream where the foward propagation is executed
   */
  template <typename Fop>
  void forward_evaluate(const Tensor2<float>& in_tensor, Tensor2<float>& out_tensor, int device_id,
                        Fop fop, cudaStream_t stream) {
    CudaDeviceContext context(device_id);

    const float* in = in_tensor.get_ptr();
    float* out = out_tensor.get_ptr();

    const int len = in_tensor.get_num_elements();
    const int grid_size = std::min((len - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
    forward_element_wise_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(in, out, len, fop);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    HCTR_LIB_THROW(cudaGetLastError());
#endif
  }

  /**
   * A method of implementing the element-wise backward pass
   * @tparam Bop the type of simple scalar lambda operation
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this operation is handled
   * @param bop Bop lambda object to do the operation per element
   * @param stream CUDA stream where the backward propagation is executed
   */
  template <typename Bop>
  void backward_evaluate(Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor, int device_id,
                         Bop bop, cudaStream_t stream) {
    CudaDeviceContext context(device_id);

    float* d_in = in_tensor.get_ptr();
    const float* d_out = out_tensor.get_ptr();

    const int len = in_tensor.get_num_elements();
    const int grid_size = std::min((len - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
    backward_element_wise_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_out, d_in, len, bop);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    HCTR_LIB_THROW(cudaGetLastError());
#endif
  }
};

}  // namespace internal
}  // namespace HugeCTR
