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
#include <utils.hpp>

#include <cpu/layers/slice_layer_cpu.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
void slice_fprop_cpu(size_t height, size_t width, std::vector<std::pair<int, int>>& ranges,
                    size_t n_outs, T* h_in, T** h_refs) {
  int i = 0;
  for (auto& range : ranges) {
    int out_width = range.second - range.first;
    for (size_t r = 0; r < height; r++) {
      for (int c = range.first; c < range.second; c++) {
        int in_idx = r * width + c;
        int out_idx = r * out_width + c - range.first;
        h_refs[i][out_idx] = h_in[in_idx];
      }
    }
    i++;
  }
}

}  // anonymous namespace

template <typename T>
SliceLayerCPU<T>::SliceLayerCPU(const Tensor2<T>& in_tensor, Tensors2<T>& out_tensors,
                          const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                          std::vector<std::pair<int, int>>& ranges)
    : LayerCPU(), virt_w_(0), ranges_(ranges) {
  try {
    if (ranges.empty()) {
      CK_THROW_(Error_t::WrongInput, "Empty slice ranges is not allowed");
    }

    if (!out_tensors.empty()) {
      CK_THROW_(Error_t::WrongInput, "output tensor vector must be empty");
    }

    auto in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be concatenated");
    }

    size_t height = in_dims[0];
    int in_w = in_dims[1];
    int prev_min = -1;
    int prev_max = 0;
    for (auto& range : ranges) {
      int cur_min = range.first;
      int cur_max = range.second;
      if (cur_min >= cur_max) {
        CK_THROW_(Error_t::WrongInput, "Reverse range is not allowed");
      }
      if (cur_min < 0 || cur_max < 0) {
        CK_THROW_(Error_t::WrongInput, "Negative ranges cannot be allowed");
      }
      if (!(prev_min <= cur_min && prev_max <= cur_max)) {
        CK_THROW_(Error_t::WrongInput, "A range cannot be out-order nor included in another");
      }
      if (cur_min >= in_w || cur_max > in_w) {
        CK_THROW_(Error_t::WrongInput, "Ranges cannot be bigger than the input width");
      }
      size_t out_w = cur_max - cur_min;
      std::vector<size_t> out_dims = {height, out_w};
      {
        Tensor2<T> tensor;
        blobs_buff->reserve(out_dims, &tensor);
        out_tensors.push_back(tensor);
      }
      sts_.push_back(cur_min);
      virt_w_ += out_w;

      prev_min = cur_min;
      prev_max = cur_max;
    }

    in_tensors_.push_back(in_tensor);
    for (auto& out_tensor : out_tensors) {
      out_tensors_.push_back(out_tensor);
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void SliceLayerCPU<T>::fprop(bool is_train) {
  T* in = in_tensors_[0].get_ptr();
  size_t n_out_tensors = out_tensors_.size();
  std::vector<T*> out;
  for (auto out_tensor : out_tensors_) {
    out.push_back(out_tensor.get_ptr());
  }
  size_t height = in_tensors_[0].get_dimensions()[0];
  size_t width = in_tensors_[0].get_dimensions()[1];
  slice_fprop_cpu(height, width, ranges_, n_out_tensors, in, out.data());
}

template <typename T>
void SliceLayerCPU<T>::bprop() {}

template class SliceLayerCPU<float>;
template class SliceLayerCPU<__half>;

}  // namespace HugeCTR
