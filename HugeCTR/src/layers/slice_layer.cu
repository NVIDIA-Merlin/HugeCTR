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
#include <cstdint>
#include <layers/slice_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
__global__ void slice_fwd_kernel(T* const __restrict__ out, const T* const __restrict__ in,
                                 const int2 in_dim, const int2 slice) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < slice.y; ni += blockDim.x) {
      out[mi * slice.y + ni] = in[mi * in_dim.y + slice.x + ni];
    }
  }
}

template <typename T>
__global__ void slice_bwd_kernel(const T* const __restrict__ out, T* const __restrict__ in,
                                 const int2 in_dim, const int2 slice) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < slice.y; ni += blockDim.x) {
      in[mi * in_dim.y + slice.x + ni] += out[mi * slice.y + ni];
    }
  }
}

}  // anonymous namespace

template <typename T>
SliceLayer<T>::SliceLayer(const Tensor2<T>& in_tensor, Tensors2<T>& out_tensors,
                          const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                          std::vector<std::pair<int, int>>& ranges,
                          const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    if (ranges.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty slice ranges is not allowed");
    }

    if (!out_tensors.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "output tensor vector must be empty");
    }

    auto in_dims = in_tensor.get_dimensions();
    if (in_dims.size() != 2) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D tensors can be concatenated");
    }

    size_t height = in_dims[0];
    int in_w = in_dims[1];
    int prev_min = -1;
    int prev_max = 0;
    for (auto& range : ranges) {
      int cur_min = range.first;
      int cur_max = range.second;
      if (cur_min >= cur_max) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Reverse range is not allowed");
      }
      if (cur_min < 0 || cur_max < 0) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Negative ranges cannot be allowed");
      }
      if (!(prev_min <= cur_min && prev_max <= cur_max)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "A range cannot be out-order nor included in another");
      }
      if (cur_min >= in_w || cur_max > in_w) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Ranges cannot be bigger than the input width");
      }
      size_t out_w = cur_max - cur_min;
      std::vector<size_t> out_dims = {height, out_w};
      {
        Tensor2<T> tensor;
        blobs_buff->reserve(out_dims, &tensor);
        out_tensors.push_back(tensor);
        out_tensors_.push_back(tensor);
      }
      slices_start_.push_back(cur_min);

      prev_min = cur_min;
      prev_max = cur_max;
    }

    in_tensor_ = in_tensor;

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void SliceLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 4;
  T* in = in_tensor_.get_ptr();
  int2 in_dim = {static_cast<int>(in_tensor_.get_dimensions()[0]),
                 static_cast<int>(in_tensor_.get_dimensions()[1])};
  int grid_size = std::min(in_dim.x, n_blocks);
  for (std::size_t i = 0; i < out_tensors_.size(); i++) {
    Tensor2<T>& out_tensor = out_tensors_[i];
    T* out = out_tensor.get_ptr();
    int2 slice = {slices_start_[i], static_cast<int>(out_tensor.get_dimensions()[1])};

    slice_fwd_kernel<<<grid_size, block_size, 0, stream>>>(out, in, in_dim, slice);
  }
}

template <typename T>
void SliceLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 4;
  T* in = in_tensor_.get_ptr();
  int2 in_dim = {static_cast<int>(in_tensor_.get_dimensions()[0]),
                 static_cast<int>(in_tensor_.get_dimensions()[1])};
  int grid_size = std::min(in_dim.x, n_blocks);
  initialize_array<<<n_blocks, block_size, 0, stream>>>(in, in_dim.x * in_dim.y, T(0));
  for (std::size_t i = 0; i < out_tensors_.size(); i++) {
    Tensor2<T>& out_tensor = out_tensors_[i];
    T* out = out_tensor.get_ptr();
    int2 slice = {slices_start_[i], static_cast<int>(out_tensor.get_dimensions()[1])};

    slice_bwd_kernel<<<grid_size, block_size, 0, stream>>>(out, in, in_dim, slice);
  }
}

template class SliceLayer<float>;
template class SliceLayer<__half>;

}  // namespace HugeCTR
