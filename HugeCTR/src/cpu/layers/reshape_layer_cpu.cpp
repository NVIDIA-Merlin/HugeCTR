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
#include <cpu/layers/reshape_layer_cpu.hpp>
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T> 
void reshape_fprop_cpu(int batch_size, int n_slot, int vector_length, size_t num_elements,
                      std::vector<int> selected, T* h_in, T* h_ref) {
  int n_active_slot = selected.empty() ? n_slot : int(selected.size());
  if (selected.empty()) {
    for (size_t i = 0; i < num_elements; i++) {
      h_ref[i] = h_in[i];
    }
  } else {
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < n_active_slot; j++) {
        for (int k = 0; k < vector_length; k++) {
          int in_idx = i * (n_slot * vector_length) + selected[j] * vector_length + k;
          int out_idx = i * (n_active_slot * vector_length) + j * vector_length + k;
          h_ref[out_idx] = h_in[in_idx];
        }
      }
    }
  }
}

}  // anonymous namespace

template <typename T>
ReshapeLayerCPU<T>::ReshapeLayerCPU(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                              size_t leading_dim)
    : LayerCPU(),
      in_place_(true),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(0) {
  try {
    const std::vector<size_t>& in_dims = in_tensor.get_dimensions();
    int im_idx = in_dims.size() - 1;
    if (leading_dim < in_dims[im_idx] || leading_dim % in_dims[im_idx] != 0) {
      CK_THROW_(Error_t::WrongInput,
                "leading_dim < in_dims[im_idx] or leading_dim % in_dims[2] != 0");
    }

    size_t n_in_elems = in_tensor.get_num_elements();
    if (leading_dim > n_in_elems) {
      CK_THROW_(Error_t::WrongInput, "leading_dim cannot be bigger than n_in_elems");
    }

    if (n_in_elems % leading_dim != 0) {
      CK_THROW_(Error_t::WrongInput, "n_in_elems % leading_dim != 0");
    }

    size_t trailing_dim = n_in_elems / leading_dim;
    std::vector<size_t> out_dims = {trailing_dim, leading_dim};

    blobs_buff->reserve(out_dims, &out_tensor);

    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
ReshapeLayerCPU<T>::ReshapeLayerCPU(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff,
                              std::vector<int>& selected)
    : LayerCPU(),
      in_place_(selected.empty()),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(selected.size()),
      selected_(selected) {
  try {
    const std::vector<size_t>& in_dims = in_tensor.get_dimensions();
    if (in_dims[1] < n_active_slot_) {
      CK_THROW_(Error_t::WrongInput, "selected is invalid");
    }

    size_t in_dims_1 = selected.empty() ? in_dims[1] : n_active_slot_;
    std::vector<size_t> out_dims = {in_dims[0], in_dims_1 * in_dims[2]};
    blobs_buff->reserve(out_dims, &out_tensor);

    if (!in_place_) {
      unsigned int i = 0;
      for (; i < in_dims.size() - 2; i++) batch_size_ += in_dims[i];
      n_slot_ = in_dims[i++];
      vector_length_ = in_dims[i];

      blobs_buff->reserve({n_active_slot_}, &selected_tensor_);
    }
    in_tensors_.push_back(in_tensor);
    out_tensors_.push_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ReshapeLayerCPU<T>::fprop(bool is_train) {
  T* h_in = in_tensors_[0].get_ptr();
  T* h_out = out_tensors_[0].get_ptr();
  size_t num_elements = in_tensors_[0].get_num_elements();
  if (in_place_) {
    for (size_t i = 0; i < num_elements; i++) {
      h_out[i] = h_in[i];
    }
  } else {
    reshape_fprop_cpu(batch_size_, n_slot_, vector_length_, num_elements, selected_, h_in, h_out);
  }
}

template <typename T>
void ReshapeLayerCPU<T>::bprop() {}

template class ReshapeLayerCPU<float>;
template class ReshapeLayerCPU<__half>;

}  // namespace HugeCTR
