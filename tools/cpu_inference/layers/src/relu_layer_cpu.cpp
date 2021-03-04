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

#include "tools/cpu_inference/layers/include/relu_layer_cpu.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void relu_cpu(T* top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > T(0.)) {
      top[i] = bottom[i];
    } else {
      top[i] = T(0.);
    }
  }
}

template <typename T>
void relu_bprop_cpu(T* d_bottom, const T* d_top, const T* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    if (bottom[i] > T(0.)) {
      d_bottom[i] = d_top[i];
    } else {
      d_bottom[i] = T(0.);
    }
  }
}

}  // namespace

template <typename T>
ReluLayerCPU<T>::ReluLayerCPU(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor)
    : LayerCPU() {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(in_tensor.get_num_elements() % 2 == 0);

  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(out_tensor);
}

template <typename T>
void ReluLayerCPU<T>::fprop(bool is_train) {

  int len = in_tensors_[0].get_num_elements();

  relu_cpu<T>(out_tensors_[0].get_ptr(), in_tensors_[0].get_ptr(), len);
}

template <typename T>
void ReluLayerCPU<T>::bprop() {}

template class ReluLayerCPU<float>;
template class ReluLayerCPU<__half>;

}  // namespace HugeCTR
