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

#include <common.hpp>
#include <cpu/layers/concat_layer_cpu.hpp>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void concat_cpu(T** input, T* output, size_t height, size_t new_width, int n_ins,
                const std::vector<size_t>& widths) {
  for (size_t r = 0; r < height; r++) {
    for (size_t c = 0; c < new_width; c++) {
      int out_idx = r * new_width + c;
      int in_no = 0;
      int c2 = c;
      size_t accum_width = 0;
      for (int k = 0; k < n_ins; k++) {
        if (c < accum_width + widths[k]) {
          in_no = k;
          c2 -= accum_width;
          break;
        }
        accum_width += widths[k];
      }
      int in_idx = r * widths[in_no] + c2;
      output[out_idx] = input[in_no][in_idx];
    }
  }
}

}  // anonymous namespace

template <typename T>
ConcatLayerCPU<T>::ConcatLayerCPU(const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
                                  const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff)
    : LayerCPU() {
  try {
    if (in_tensors.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty input tensors");
    }

    int n_in_tensors = in_tensors.size();
    size_t height = 0;
    size_t new_width = 0;
    for (int i = 0; i < n_in_tensors; i++) {
      auto cur_in_dims = in_tensors[i].get_dimensions();
      if (i != 0) {
        auto first_in_dims = in_tensors[0].get_dimensions();
        if (cur_in_dims[0] != first_in_dims[0]) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same height");
        }
      }
      if (cur_in_dims.size() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D tensors can be concatenated");
      }
      if (i == 0) {
        height = cur_in_dims[0];
      }
      new_width += cur_in_dims[1];
    }

    std::vector<size_t> out_dims = {height, new_width};
    blobs_buff->reserve(out_dims, &out_tensor);

    for (const Tensor2<T>& in_tensor : in_tensors) {
      in_tensors_.push_back(in_tensor);
    }
    out_tensor_ = out_tensor;

    blobs_buff->reserve({in_tensors.size()}, &h_inputs_);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ConcatLayerCPU<T>::fprop(bool is_train) {
  size_t height = out_tensor_.get_dimensions()[0];
  int n_ins = in_tensors_.size();
  std::vector<size_t> widths;
  size_t new_width = 0;
  for (const Tensor2<T>& in_tensor : in_tensors_) {
    widths.push_back(in_tensor.get_dimensions()[1]);
    new_width += in_tensor.get_dimensions()[1];
  }
  // fprop
  T* output = out_tensor_.get_ptr();
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    h_inputs_.get_ptr()[i] = in_tensors_[i].get_ptr();
  }
  concat_cpu(h_inputs_.get_ptr(), output, height, new_width, n_ins, widths);
}

template <typename T>
void ConcatLayerCPU<T>::bprop() {}

template class ConcatLayerCPU<float>;
template class ConcatLayerCPU<__half>;

}  // namespace HugeCTR
