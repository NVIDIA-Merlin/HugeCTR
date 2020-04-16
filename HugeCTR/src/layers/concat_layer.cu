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

#include "HugeCTR/include/layers/concat_layer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor.hpp"

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
__global__ void concat_kernel(bool forward, T* out, const int h, const int out_w,
                              const Args... args) {
  const int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const ConcatLayer::InParam<T> in_params[] = {args...};
  const int n_ins = array_length(in_params);
  for (int gid = gid_base; gid < h * out_w; gid += blockDim.x * gridDim.x) {
    int row = gid / out_w;
    int out_col = gid % out_w;
    int out_idx = row * out_w + out_col;

    int in_no = 0;
    int in_col = out_col;
    int accum_width = 0;
    for (int k = 0; k < n_ins; k++) {
      if (out_col < accum_width + in_params[k].in_w) {
        in_no = k;
        in_col -= accum_width;
        break;
      }
      accum_width += in_params[k].in_w;
    }
    T* in = in_params[in_no].in;
    int in_idx = row * in_params[in_no].in_w + in_col;

    if (forward) {
      out[out_idx] = in[in_idx];
    } else {
      in[in_idx] = out[out_idx];
    }
  }
}

}  // anonymous namespace

ConcatLayer::ConcatLayer(Tensors<float>& in_tensors, std::shared_ptr<Tensor<float>>& out_tensor,
                         const std::shared_ptr<GeneralBuffer<float>>& blobs_buff, int device_id)
    : Layer(device_id), n_sms_(0) {
  try {
    CudaDeviceContext context(get_device_id());

    if (in_tensors.empty()) {
      CK_THROW_(Error_t::WrongInput, "Empty input tensors");
    }

    int n_in_tensors = in_tensors.size();
    size_t height = 0;
    size_t new_width = 0;
    for (int i = 0; i < n_in_tensors; i++) {
      auto cur_in_dims = in_tensors[i]->get_dims();
      if (i != 0) {
        auto first_in_dims = in_tensors[0]->get_dims();
        if (cur_in_dims[0] != first_in_dims[0]) {
          CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same height");
        }
      }
      if (cur_in_dims.size() != 2) {
        CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be concatenated");
      }
      if (in_tensors[i]->get_format() != TensorFormat_t::HW) {
        CK_THROW_(Error_t::WrongInput, "Only TensorFormat_t::HW is allowed");
      }
      if (i == 0) {
        height = cur_in_dims[0];
      }
      new_width += cur_in_dims[1];
    }

    std::vector<size_t> out_dims = {height, new_width};
    out_tensor.reset(new Tensor<float>(out_dims, blobs_buff, TensorFormat_t::HW));

    for (auto& in_tensor : in_tensors) {
      in_tensors_.emplace_back(in_tensor);
    }
    out_tensors_.emplace_back(out_tensor);

    int device = get_device_id();
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sms_, cudaDevAttrMultiProcessorCount, device));
    assert(n_sms_ > 0);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void ConcatLayer::fprop(cudaStream_t stream) { prop_common(true, stream); }

void ConcatLayer::bprop(cudaStream_t stream) { prop_common(false, stream); }

void ConcatLayer::prop_common(bool forward, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  int n_in_tensors = in_tensors_.size();
  if (n_in_tensors == 2) {
    std::vector<InParam<float>> in_params = set_in_params(2);
    kernel_launch(forward, stream, in_params[0], in_params[1]);
  } else if (n_in_tensors == 3) {
    std::vector<InParam<float>> in_params = set_in_params(3);
    kernel_launch(forward, stream, in_params[0], in_params[1], in_params[2]);
  } else if (n_in_tensors == 4) {
    std::vector<InParam<float>> in_params = set_in_params(4);
    kernel_launch(forward, stream, in_params[0], in_params[1], in_params[2], in_params[3]);
  } else if (n_in_tensors == 5) {
    std::vector<InParam<float>> in_params = set_in_params(5);
    kernel_launch(forward, stream, in_params[0], in_params[1], in_params[2], in_params[3],
                  in_params[4]);
  } else {
    CK_THROW_(Error_t::UnSupportedFormat, "Merging > 5 layers is not supported");
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

std::vector<ConcatLayer::InParam<float>> ConcatLayer::set_in_params(int n) {
  std::vector<InParam<float>> in_params;
  for (int i = 0; i < n; i++) {
    const auto& in_tensor = in_tensors_[i];
    float* in = in_tensor->get_ptr();
    int w = in_tensor->get_dims()[1];
    in_params.push_back({in, w});
  }
  return std::move(in_params);
}

template <typename... Args>
void ConcatLayer::kernel_launch(bool forward, cudaStream_t stream, Args&... args) {
  int block_size = 256;
  int n_blocks = n_sms_ * 8;
  const auto& out_tensor = out_tensors_[0];
  float* out = out_tensor->get_ptr();
  int h = out_tensor->get_dims()[0];
  int out_w = out_tensor->get_dims()[1];
  concat_kernel<<<n_blocks, block_size, 0, stream>>>(forward, out, h, out_w, args...);
}

}  // namespace HugeCTR
