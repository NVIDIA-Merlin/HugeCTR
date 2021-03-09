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

#pragma once

#include <algorithm>
#include <functional>
#include <tensor2.hpp>
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {
namespace internal {

template <typename Fop>
void forward_element_wise_cpu(const float* in, float* out, int len, Fop fop) {
  for (int i = 0; i < len; i++) {
    out[i] = fop(in[i]);
  }
}

template <typename Bop>
void backward_element_wise_cpu(const float* h_out, float* h_in, int len, Bop bop) {
  for (int i = 0; i < len; i++) {
    h_in[i] = bop(h_out[i], h_in[i]);
  }
}

/**
 * Common implementation for the element wise layers such as Relu and Elu.
 * Their fprop/brop are just the wrapperw of forward_evaluate/backward_evaluate,
 * while passing the simple scalar lambda operations to them.
 * All the other element wise layers can be implementated in the similar way.
 */
class ElementWiseFunctorCPU {
 public:
  /**
   * Ctor of ElementWiseFunctor. Copy construction and assigment are disabled.
   */
  ElementWiseFunctorCPU() {}
  ElementWiseFunctorCPU(const ElementWiseFunctorCPU&) = delete;
  ElementWiseFunctorCPU& operator=(const ElementWiseFunctorCPU&) = delete;

  /**
   * D'tor of ElementWiseFunctor.
   */
  ~ElementWiseFunctorCPU() {}

  /**
   * A method of implementing the element-wise forward pass
   * @tparam Fop the type of simple scalar lambda operation
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this operation is handled
   * @param fop Fop lambda object to do the operation per element
   */
  template <typename Fop>
  void forward_evaluate(const Tensor2<float>& in_tensor, Tensor2<float>& out_tensor, int device_id,
                        Fop fop) {
    const float* in = in_tensor.get_ptr();
    float* out = out_tensor.get_ptr();

    const int len = in_tensor.get_num_elements();
    forward_element_wise_cpu(in, out, len, fop);
  }

  /**
   * A method of implementing the element-wise backward pass
   * @tparam Bop the type of simple scalar lambda operation
   * @param in_tensor the input tensor
   * @param out_tensor the output tensor which has the same dim with in_tensor
   * @param device_id the id of GPU where this operation is handled
   * @param bop Bop lambda object to do the operation per element
   */
  template <typename Bop>
  void backward_evaluate(Tensor2<float>& in_tensor, const Tensor2<float>& out_tensor, int device_id,
                         Bop bop) {
    float* h_in = in_tensor.get_ptr();
    const float* h_out = out_tensor.get_ptr();

    const int len = in_tensor.get_num_elements();
    forward_element_wise_cpu(h_out, h_in, len, bop);
  }
};

}  // namespace internal
}  // namespace HugeCTR
