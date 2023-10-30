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

#include <common.hpp>
#include <cstdint>
#include <layers/slice_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.cuh>
#include <utils.hpp>
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
SliceLayer<T>::SliceLayer(const core23::Tensor& input_tensor,
                          std::vector<core23::Tensor>& output_tensors,
                          std::vector<std::pair<int, int>>& ranges,
                          const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {}, gpu_resource) {
  try {
    if (ranges.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty slice ranges is not allowed");
    }

    if (!output_tensors.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "output tensor vector must be empty");
    }
    auto in_shape = input_tensor.shape();
    auto dims = in_shape.dims();

    int in_w = in_shape.size(dims - 1);
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
      core23::Shape out_shape(dims);
      for (auto i = 0; i < dims - 1; i++) {
        out_shape.set(i, in_shape.size(i));
      }
      out_shape.set(dims - 1, cur_max - cur_min);
      core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};
      core23::Tensor tensor(input_tensor.my_params().shape(out_shape).buffer_params(buf_p));
      output_tensors.push_back(tensor);
      slices_start_.push_back(cur_min);

      prev_min = cur_min;
      prev_max = cur_max;
    }
    output_tensors_ = output_tensors;

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
  T* in = input_tensors_[0].data<T>();
  auto in_shape = input_tensors_[0].shape();
  auto dims = in_shape.dims();
  int height = 1;
  for (auto i = 0; i < dims - 1; i++) {
    height = height * in_shape.size(i);
  }
  int width = in_shape.size(dims - 1);
  int2 in_dim = {static_cast<int>(height), static_cast<int>(width)};
  int grid_size = std::min(in_dim.x, n_blocks);
  for (std::size_t i = 0; i < output_tensors_.size(); i++) {
    core23::Tensor& output_tensor = output_tensors_[i];
    T* out = output_tensor.data<T>();
    int2 slice = {slices_start_[i], static_cast<int>(output_tensor.size(dims - 1))};

    slice_fwd_kernel<<<grid_size, block_size, 0, stream>>>(out, in, in_dim, slice);
  }
}

template <typename T>
void SliceLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 4;
  T* in = input_tensors_[0].data<T>();
  auto in_shape = input_tensors_[0].shape();
  auto dims = in_shape.dims();
  int height = 1;
  for (auto i = 0; i < dims - 1; i++) {
    height = height * in_shape.size(i);
  }
  int width = in_shape.size(dims - 1);
  int2 in_dim = {static_cast<int>(height), static_cast<int>(width)};
  int grid_size = std::min(in_dim.x, n_blocks);
  initialize_array<<<n_blocks, block_size, 0, stream>>>(in, in_dim.x * in_dim.y, T(0));
  for (std::size_t i = 0; i < output_tensors_.size(); i++) {
    core23::Tensor& output_tensor = output_tensors_[i];
    T* out = output_tensor.data<T>();
    int2 slice = {slices_start_[i], static_cast<int>(output_tensor.size(dims - 1))};

    slice_bwd_kernel<<<grid_size, block_size, 0, stream>>>(out, in, in_dim, slice);
  }
}

template class SliceLayer<float>;
template class SliceLayer<__half>;

}  // namespace HugeCTR
