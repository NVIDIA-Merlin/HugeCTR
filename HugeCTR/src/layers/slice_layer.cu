/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/layers/slice_layer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <size_t length, typename T>
__device__ int array_length(T (&arr)[length]) { return length; }

template <typename T, typename... Args>
__global__ void slice_kernel(bool forward, T* in, const int h, const int in_w, const int virt_w, const Args... args) {
  const int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const SliceLayer::OutParam<T> out_params[] = {args...};
  const int n_outs = array_length(out_params);

  for(int gid = gid_base; gid < h * virt_w; gid+= blockDim.x * gridDim.x) {
    int row = gid / virt_w;
    int virt_col = gid % virt_w;

    int in_col = 0;
    T* out = nullptr;
    int out_w = 0;
    int out_col = 0;

    int accum_margin = 0;
    int prev_ed = 0;
    for(int k = 0; k < n_outs; k++) {
      int st = out_params[k].st;
      int ed = out_params[k].ed;
      accum_margin += (st - prev_ed);
      int cand_in_col = virt_col + accum_margin;
      if(cand_in_col >= st && cand_in_col < ed) {
        in_col = cand_in_col;
        out = out_params[k].out;
        out_w = ed - st;
        out_col = in_col - st;
        break;
      }
      prev_ed = ed;
    }

    int in_idx = row * in_w + in_col;
    int out_idx = row * out_w + out_col;

    if(forward) {
      out[out_idx] = in[in_idx];
    }
    else {
      in[in_idx] = out[out_idx];
    }
  }
}

}  // anonymous namespace

SliceLayer::SliceLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
                       Tensors<float>& out_tensors,
                       const std::shared_ptr<GeneralBuffer<float>>& blobs_buff,
                       std::set<std::pair<int,int>>& ranges,
                       int device_id)
    : Layer(device_id),
      n_sms_(0),
      virt_w_(0) {
  try {
    CudaDeviceContext context(device_id);

    if(ranges.empty()) {
      CK_THROW_(Error_t::WrongInput, "Empty slice ranges is not allowed");
    }

    if(!out_tensors.empty()) {
      CK_THROW_(Error_t::WrongInput, "output tensor vector must be empty");
    }

    auto in_dims = in_tensor->get_dims();
    if(in_dims.size() != 2) {
      CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be concatenated");
    }
    if(in_tensor->get_format() != TensorFormat_t::HW) {
      CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed");
    }

    int height = in_dims[0];
    int in_w = in_dims[1];
    int prev_max = 0;
    for(auto& range : ranges) {
      int cur_min = range.first;
      int cur_max = range.second;
      if(cur_min >= cur_max) {
        CK_THROW_(Error_t::WrongInput, "Reverse range is not allowed");
      }
      if(cur_min < prev_max || cur_max < prev_max) {
        CK_THROW_(Error_t::WrongInput, "Ranges cannot be overlapped");
      }
      if(cur_min >= in_w || cur_max > in_w) {
        CK_THROW_(Error_t::WrongInput, "Ranges cannot be bigger than the input width");
      }
      int out_w = cur_max - cur_min;
      std::vector<int> out_dims = {height, out_w};
      out_tensors.emplace_back(new Tensor<float>(out_dims, blobs_buff, TensorFormat_t::HW));
      sts_.push_back(cur_min);
      virt_w_ += out_w;

      prev_max = cur_max;
    }

    in_tensors_.emplace_back(in_tensor);
    for(auto out_tensor : out_tensors) {
      out_tensors_.emplace_back(out_tensor);
    }

    int device = get_device_id();
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sms_, cudaDevAttrMultiProcessorCount, device));
    assert(n_sms_ > 0);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void SliceLayer::fprop(cudaStream_t stream) {
  prop_common(true, stream);
}

void SliceLayer::bprop(cudaStream_t stream) {
  prop_common(false, stream);
}

void SliceLayer::prop_common(bool forward, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  int n_out_tensors = out_tensors_.size();
  if(n_out_tensors == 2) {
    std::vector<OutParam<float>> out_params = set_out_params(2);
    kernel_launch(forward, stream, out_params[0], out_params[1]);
  }
  else if(n_out_tensors == 3) {
    std::vector<OutParam<float>> out_params = set_out_params(3);
    kernel_launch(forward, stream, out_params[0], out_params[1], out_params[2]);
  }
  else if(n_out_tensors == 4) {
    std::vector<OutParam<float>> out_params = set_out_params(4);
    kernel_launch(forward, stream, out_params[0], out_params[1], out_params[2], out_params[3]);
  }
  else if(n_out_tensors == 5) {
    std::vector<OutParam<float>> out_params = set_out_params(5);
    kernel_launch(forward, stream, out_params[0], out_params[1], out_params[2], out_params[3], out_params[4]);
  }
  else {
    CK_THROW_(Error_t::UnSupportedFormat, "Merging > 5 layers is not supported");
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

std::vector<SliceLayer::OutParam<float>> SliceLayer::set_out_params(int n) {
  std::vector<OutParam<float>> out_params;
  for(int i = 0; i < n; i++) {
    const auto& out_tensor = out_tensors_[i];
    float* out = out_tensor->get_ptr();
    int st = sts_[i];
    int w = out_tensor->get_dims()[1];
    out_params.push_back({out, st, st + w});
  }
  return std::move(out_params);
}

template <typename... Args>
void SliceLayer::kernel_launch(bool forward, cudaStream_t stream, Args&... args) {
  int block_size = 256;
  int n_blocks = n_sms_ * 8;
  const auto& in_tensor = in_tensors_[0];
  float* in = in_tensor->get_ptr();
  int h = in_tensor->get_dims()[0];
  int in_w = in_tensor->get_dims()[1];
  slice_kernel<<<n_blocks, block_size, 0, stream>>>(forward, in, h, in_w, virt_w_, args...);
}

}  // namespace HugeCTR
