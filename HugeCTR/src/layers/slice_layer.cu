/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <layers/slice_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <size_t length, typename T>
__device__ int array_length(T (&arr)[length]) {
  return length;
}

template <typename T, typename... Args>
__global__ void slice_kernel(bool forward, T* in, const int h, const int in_w, const int virt_w,
                             const Args... args) {
  const typename SliceLayer<T>::OutParam out_params[] = {args...};
  const int n_outs = array_length(out_params);

  for (int row = blockIdx.x; row < h; row += gridDim.x) {
    for (int k = 0; k < n_outs; k++) {
      int st = out_params[k].st;
      int ed = out_params[k].ed;
      int out_w = ed - st;
      for (int out_col = threadIdx.x; out_col < out_w; out_col += blockDim.x) {
        int in_col = out_col + st;
        int in_idx = row * in_w + in_col;
        int out_idx = row * out_w + out_col;
        T* out = out_params[k].out;
        if (forward) {
          out[out_idx] = in[in_idx];
        } else {
          in[in_idx] += out[out_idx];
        }
      }
      __syncthreads();
    }
  }
}

}  // anonymous namespace

template <typename T>
SliceLayer<T>::SliceLayer(const Tensor2<T>& in_tensor, Tensors2<T>& out_tensors,
                          const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                          std::vector<std::pair<int, int>>& ranges,
                          const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), virt_w_(0) {
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
void SliceLayer<T>::fprop(bool is_train) {
  prop_common(true, is_train, get_gpu().get_stream());
}

template <typename T>
void SliceLayer<T>::bprop() {
  prop_common(false, true, get_gpu().get_stream());
}

template <typename T>
void SliceLayer<T>::prop_common(bool forward, bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  int n_out_tensors = out_tensors_.size();
  if (n_out_tensors == 2) {
    std::vector<OutParam> out_params = set_out_params(2);
    kernel_launch(forward, is_train, stream, out_params[0], out_params[1]);
  } else if (n_out_tensors == 3) {
    std::vector<OutParam> out_params = set_out_params(3);
    kernel_launch(forward, is_train, stream, out_params[0], out_params[1], out_params[2]);
  } else if (n_out_tensors == 4) {
    std::vector<OutParam> out_params = set_out_params(4);
    kernel_launch(forward, is_train, stream, out_params[0], out_params[1], out_params[2],
                  out_params[3]);
  } else if (n_out_tensors == 5) {
    std::vector<OutParam> out_params = set_out_params(5);
    kernel_launch(forward, is_train, stream, out_params[0], out_params[1], out_params[2],
                  out_params[3], out_params[4]);
  } else {
    CK_THROW_(Error_t::UnSupportedFormat, "Slicing into > 5 layers is not supported");
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
std::vector<typename SliceLayer<T>::OutParam> SliceLayer<T>::set_out_params(int n) {
  std::vector<OutParam> out_params;
  for (int i = 0; i < n; i++) {
    Tensor2<T>& out_tensor = out_tensors_[i];
    T* out = out_tensor.get_ptr();
    int st = sts_[i];
    int w = out_tensor.get_dimensions()[1];
    out_params.push_back({out, st, st + w});
  }
  return std::move(out_params);
}

template <typename T>
template <typename... Args>
void SliceLayer<T>::kernel_launch(bool forward, bool is_train, cudaStream_t stream, Args&... args) {
  int block_size = 512;
  int n_blocks = get_gpu().get_sm_count() * 4;
  Tensor2<T>& in_tensor = get_in_tensors(is_train)[0];
  T* in = in_tensor.get_ptr();
  int h = in_tensor.get_dimensions()[0];
  int in_w = in_tensor.get_dimensions()[1];
  if (!forward) {
    initialize_array<<<n_blocks, block_size, 0, stream>>>(in, h * in_w, T(0));
  }
  slice_kernel<<<n_blocks, block_size, 0, stream>>>(forward, in, h, in_w, virt_w_, args...);
}

template class SliceLayer<float>;
template class SliceLayer<__half>;

}  // namespace HugeCTR
