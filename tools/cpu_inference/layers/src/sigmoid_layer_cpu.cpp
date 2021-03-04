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
#include <utils.hpp>

#include "tools/cpu_inference/layers/include/sigmoid_layer_cpu.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void sigmoid_cpu(T* top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = T(1.) / ( T(1.) + exp(-bottom[i]) );
  }
}

template<>
void sigmoid_cpu(__half* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(1.0 / ( 1.0 + exp(-__half2float(bottom[i])) ) );
  }
}

template <typename T>
void sigmoid_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    T y = T(1.) / ( T(1.) + exp(-bottom[i]) );
    d_bottom[i] = d_top[i] * y * (T(1.)-y);
  }
}

template<>
void sigmoid_bprop_cpu(__half* d_bottom, const __half* d_top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
      float y = 1.0 / ( 1.0 + exp(-__half2float(bottom[i])) );
      d_bottom[i] = __float2half(__half2float(d_top[i]) * y * (1.0 - y));
  }
}

} // end namespace

template <typename T>
SigmoidLayerCPU<T>::SigmoidLayerCPU(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor)
    : LayerCPU() {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(in_tensor.get_num_elements() % 2 == 0);

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void SigmoidLayerCPU<T>::fprop(bool is_train) {

  int len = in_tensors_[0].get_num_elements();

  sigmoid_cpu<T>(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), len);
}

template <typename T>
void SigmoidLayerCPU<T>::bprop() {}

template class SigmoidLayerCPU<float>;
template class SigmoidLayerCPU<__half>;

}  // namespace HugeCTR
