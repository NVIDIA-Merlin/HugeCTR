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

#include <cpu/layers/cast_layer_cpu.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

void cast_cpu(__half* top, const float* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __float2half(bottom[i]);
  }
}

void cast_cpu(float* top, const __half* bottom, int len) {
  for (int i = 0; i < len; ++i) {
    top[i] = __half2float(bottom[i]);
  }
}

}  // namespace

template <typename From, typename To>
CastLayerCPU<From, To>::CastLayerCPU(const Tensor2<From>& bottom_tensor,
                                     const Tensor2<To>& top_tensor)
    : LayerCPU() {
  assert(bottom_tensor.get_num_elements() == top_tensor.get_num_elements());

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

template <typename From, typename To>
void CastLayerCPU<From, To>::fprop(bool is_train) {
  const From* bottom = bottom_tensor_.get_ptr();
  To* top = top_tensor_.get_ptr();
  int len = bottom_tensor_.get_num_elements();
  cast_cpu(top, bottom, len);
}

template <typename From, typename To>
void CastLayerCPU<From, To>::bprop() {}

template class CastLayerCPU<float, __half>;
template class CastLayerCPU<__half, float>;

}  // namespace HugeCTR
