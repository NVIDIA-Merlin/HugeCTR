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

#include <HugeCTR/include/utils.hpp>
#include <algorithm>
#include <cpu/layers/dropout_layer_cpu.hpp>
#include <cstdio>
#include <ctime>
#include <data_generator.hpp>
#include <functional>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void dropout_cpu(T* input, T* output, float* mask, float rate, float scale, size_t num,
                 bool is_train) {
  for (size_t i = 0; i < num; i++) {
    output[i] = is_train ? ((1.f - mask[i]) >= rate) * input[i] * scale : input[i];
  }
}

template <>
void dropout_cpu(__half* input, __half* output, float* mask, float rate, float scale, size_t num,
                 bool is_train) {
  for (size_t i = 0; i < num; i++) {
    output[i] = is_train ? __float2half(((1.f - mask[i]) >= rate) * __half2float(input[i]) * scale)
                         : input[i];
  }
}

}  // end namespace

template <typename T>
DropoutLayerCPU<T>::DropoutLayerCPU(const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor,
                                    const std::shared_ptr<GeneralBuffer2<HostAllocator>> blobs_buff,
                                    float rate)
    : LayerCPU(), rate_(rate), scale_(1.0 / (1.0 - rate + 1e-6)) {
  assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(rate_ > 0.f && rate_ < 1.f);

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);

  blobs_buff->reserve(in_tensor.get_dimensions(), &mask_);
}

template <typename T>
void DropoutLayerCPU<T>::fprop(bool is_train) {
  FloatUniformDataSimulator<float> ldata_sim(0.f, 1.f);
  size_t num = 1;
  for (auto dim : in_tensors_[0].get_dimensions()) {
    num *= dim;
  }
  float* h_mask = mask_.get_ptr();
  for (size_t i = 0; i < num; i++) {
    h_mask[i] = ldata_sim.get_num();
  }
  T* input = in_tensors_[0].get_ptr();
  T* output = out_tensors_[0].get_ptr();
  dropout_cpu(input, output, h_mask, rate_, scale_, num, is_train);
}

template <typename T>
void DropoutLayerCPU<T>::bprop() {}

template class DropoutLayerCPU<float>;
template class DropoutLayerCPU<__half>;

}  // namespace HugeCTR
