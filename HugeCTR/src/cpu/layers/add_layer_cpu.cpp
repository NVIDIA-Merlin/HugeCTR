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

#include <algorithm>
#include <cpu/layers/add_layer_cpu.hpp>
#include <functional>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
void add_cpu(T** input, T* output, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += input[j][i];
    }
    output[i] = tmp;
  }
}

template <>
void add_cpu(__half** input, __half* output, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    float tmp = 0.f;
    for (size_t j = 0; j < num; j++) {
      tmp += __half2float(input[j][i]);
    }
    output[i] = __float2half(tmp);
  }
}

template <typename T>
void add_dgrad_cpu(const T* top_grad, T** dgrad, size_t size, size_t num) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      dgrad[j][i] = top_grad[i];
    }
  }
}

}  // end of namespace

template <typename T>
AddLayerCPU<T>::AddLayerCPU(const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
                            const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff)
    : LayerCPU() {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "AddLayer needs at least 2 input tensors");
    }
    for (size_t i = 1; i < num_; i++) {
      if (in_tensors[i].get_dimensions().size() != dims.size()) {
        HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.size(); j++) {
        if (in_tensors[i].get_dimensions()[j] != dims[j]) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve({num_}, &h_inputs_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void AddLayerCPU<T>::initialize() {
  for (size_t i = 0; i < num_; i++) {
    h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
  }
}

template <typename T>
void AddLayerCPU<T>::fprop(bool is_train) {
  T* output = out_tensors_[0].get_ptr();

  add_cpu(h_inputs_.get_ptr(), output, size_, num_);
}

template <>
void AddLayerCPU<__half>::fprop(bool is_train) {
  __half* output = out_tensors_[0].get_ptr();

  add_cpu(h_inputs_.get_ptr(), output, size_, num_);
}

template <typename T>
void AddLayerCPU<T>::bprop() {}

template <>
void AddLayerCPU<__half>::bprop() {}

template class AddLayerCPU<float>;
template class AddLayerCPU<__half>;

}  // namespace HugeCTR
