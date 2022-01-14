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

#include <algorithm>
#include <cpu/layers/elementwise_multiply_layer_cpu.hpp>
#include <functional>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void elementwise_multiply_cpu(T** input, T* output, size_t size, size_t num) {
  T one = 1.0;

  for (size_t i = 0; i < size; i++) {
    T tmp = one;
    for (size_t j = 0; j < num; j++) {
      tmp = tmp * input[j][i];
    }
    output[i] = tmp;
  }
}

template <typename T>
void elementwise_multiply_dgrad_cpu(const T* top_grad, T** dgrad, const T* fprop_output,
                                    size_t size, size_t num) {
  T zero = 0.0;

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < num; j++) {
      if (0 == fprop_output[i]) {
        dgrad[j][i] = zero;
      } else {
        T d_input = dgrad[j][i];
        dgrad[j][i] = top_grad[i] * T(fprop_output[i] / d_input);
      }
    }
  }
}

}  // end of namespace

template <typename T>
ElementwiseMultiplyLayerCPU<T>::ElementwiseMultiplyLayerCPU(
    const Tensors2<T>& in_tensors, const Tensor2<T>& out_tensor,
    const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff)
    : LayerCPU() {
  try {
    size_ = in_tensors[0].get_num_elements();
    num_ = in_tensors.size();

    // error input checking
    auto dims = in_tensors[0].get_dimensions();
    if (num_ < 2) {
      CK_THROW_(Error_t::WrongInput, "ElementwiseMultiplyLayer needs at least 2 input tensors");
    }
    for (size_t i = 1; i < num_; i++) {
      if (in_tensors[i].get_dimensions().size() != dims.size()) {
        CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same num of dims");
      }
      for (unsigned int j = 0; j < dims.size(); j++) {
        if (in_tensors[i].get_dimensions()[j] != dims[j]) {
          CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same dims");
        }
      }
    }

    for (size_t i = 0; i < num_; i++) {
      in_tensors_.push_back(in_tensors[i]);
    }
    out_tensors_.push_back(out_tensor);

    blobs_buff->reserve({num_}, &h_inputs_);
    blobs_buff->reserve(out_tensor.get_dimensions(), &fprop_output_);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ElementwiseMultiplyLayerCPU<T>::initialize() {
  for (size_t i = 0; i < num_; i++) {
    h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
  }
}

template <typename T>
void ElementwiseMultiplyLayerCPU<T>::fprop(bool is_train) {
  if (!initialized_) {
    for (size_t i = 0; i < num_; i++) {
      h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
    }
    initialized_ = true;
  }
  T* output = out_tensors_[0].get_ptr();
  elementwise_multiply_cpu(h_inputs_.get_ptr(), output, size_, num_);
}

template <typename T>
void ElementwiseMultiplyLayerCPU<T>::bprop() {}

template class ElementwiseMultiplyLayerCPU<float>;
template class ElementwiseMultiplyLayerCPU<__half>;

}  // namespace HugeCTR
