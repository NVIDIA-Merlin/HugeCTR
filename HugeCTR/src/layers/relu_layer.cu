/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <algorithm>
#include <functional>
#include "HugeCTR/include/utils.hpp"
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

ReluLayer::ReluLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
                     const std::shared_ptr<Tensor<float>>& out_tensor, int device_id)
    : Layer(device_id) {
  assert(get_size_from_dims(in_tensor->get_dims()) == get_size_from_dims(out_tensor->get_dims()));

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);
}

void ReluLayer::fprop(cudaStream_t stream) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  auto fop = [] __device__(float in) { return (in < 0) ? 0 : in; };
  internal::ElementWiseFunctor functor;
  functor.forward_evaluate(*in_tensor, *out_tensor, get_device_id(), fop, stream);
}

void ReluLayer::bprop(cudaStream_t stream) {
  const auto& in_tensor = in_tensors_[0];
  const auto& out_tensor = out_tensors_[0];

  auto bop = [] __device__(float d_out, float d_in) { return (d_in < 0) ? 0 : d_out; };
  internal::ElementWiseFunctor functor;
  functor.backward_evaluate(*in_tensor, *out_tensor, get_device_id(), bop, stream);
}

}  // namespace HugeCTR
