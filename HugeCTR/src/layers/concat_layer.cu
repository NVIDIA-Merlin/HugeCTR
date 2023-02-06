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
#include <layers/concat_layer.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void concat_fwd_kernel(T* out, const int2 out_dim, T* in, const int2 in_dim,
                                  int offset) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < in_dim.y; ni += blockDim.x) {
      out[mi * out_dim.y + offset + ni] = in[mi * in_dim.y + ni];
    }
  }
}

template <typename T>
__global__ void concat_bwd_kernel(T* out, const int2 out_dim, T* in, const int2 in_dim,
                                  int offset) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < in_dim.y; ni += blockDim.x) {
      in[mi * in_dim.y + ni] = out[mi * out_dim.y + offset + ni];
    }
  }
}

}  // namespace

template <typename T>
ConcatLayer<T>::ConcatLayer(const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
                            const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
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

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ConcatLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int n_in_tensors = in_tensors_.size();
  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 8;
  T* out = out_tensor_.get_ptr();
  const int2 out_dim = {static_cast<int>(out_tensor_.get_dimensions()[0]),
                        static_cast<int>(out_tensor_.get_dimensions()[1])};
  int offset = 0;
  for (int i = 0; i < n_in_tensors; i++) {
    Tensor2<T>& in_tensor = in_tensors_[i];
    T* in = in_tensor.get_ptr();
    const int2 in_dim = {static_cast<int>(in_tensor.get_dimensions()[0]),
                         static_cast<int>(in_tensor.get_dimensions()[1])};

    concat_fwd_kernel<<<n_blocks, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    offset += in_dim.y;
  }
}

template <typename T>
void ConcatLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 8;
  T* out = out_tensor_.get_ptr();
  const int2 out_dim = {static_cast<int>(out_tensor_.get_dimensions()[0]),
                        static_cast<int>(out_tensor_.get_dimensions()[1])};
  int grid_size = std::min(out_dim.x, n_blocks);
  int offset = 0;
  for (std::size_t i = 0; i < in_tensors_.size(); i++) {
    Tensor2<T>& in_tensor = in_tensors_[i];
    T* in = in_tensor.get_ptr();
    const int2 in_dim = {static_cast<int>(in_tensor.get_dimensions()[0]),
                         static_cast<int>(in_tensor.get_dimensions()[1])};

    concat_bwd_kernel<<<grid_size, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    offset += in_dim.y;
  }
}

template class ConcatLayer<float>;
template class ConcatLayer<__half>;

}  // namespace HugeCTR
