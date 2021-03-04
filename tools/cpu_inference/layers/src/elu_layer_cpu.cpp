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
#include "tools/cpu_inference/layers/include/element_wise_function_cpu.hpp"
#include "tools/cpu_inference/layers/include/elu_layer_cpu.hpp"
#include <utils.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void elu_cpu(const T* in, T* out, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    out[i] = (in[i] < 0) ? T(alpha * (exp(in[i]) - 1)) : in[i];
  }
}

template <typename T>
void elu_bprop_cpu(const T* d_out, T* d_in, int len, T alpha) {
  for (int i = 0; i < len; ++i) {
    d_in[i] = (d_in[i] < 0) ? T(alpha * exp(d_in[i]) * d_out[i]) : d_out[i];
  }
}

} // end namespace

template <typename T>
EluLayerCPU<T>::EluLayerCPU(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, T alpha)
    : LayerCPU(), alpha_(alpha) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void EluLayerCPU<T>::fprop(bool is_train) {

  const Tensor2<T>& in_tensor = in_tensors_[0];
  Tensor2<T>& out_tensor = out_tensors_[0];

  const int len = in_tensor.get_num_elements();

  T alpha = alpha_;

  elu_cpu(in_tensor.get_ptr(), out_tensor.get_ptr(), len, alpha);
}

template <typename T>
void EluLayerCPU<T>::bprop() {}

template class EluLayerCPU<float>;
template class EluLayerCPU<__half>;

}  // namespace HugeCTR
