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
#include <layers/concat_layer.hpp>
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
__global__ void concat_kernel(bool forward, T* out, const int h, const int out_w,
                              const Args... args) {
  const int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
  const typename ConcatLayer<T>::InParam in_params[] = {args...};
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

template <typename T>
ConcatLayer<T>::ConcatLayer(const Tensors2<T>& in_tensors, Tensor2<T>& out_tensor,
                            const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  try {
    if (in_tensors.empty()) {
      CK_THROW_(Error_t::WrongInput, "Empty input tensors");
    }

    int n_in_tensors = in_tensors.size();
    size_t height = 0;
    size_t new_width = 0;
    for (int i = 0; i < n_in_tensors; i++) {
      auto cur_in_dims = in_tensors[i].get_dimensions();
      if (i != 0) {
        auto first_in_dims = in_tensors[0].get_dimensions();
        if (cur_in_dims[0] != first_in_dims[0]) {
          CK_THROW_(Error_t::WrongInput, "All the input tensors must have the same height");
        }
      }
      if (cur_in_dims.size() != 2) {
        CK_THROW_(Error_t::WrongInput, "Only 2D tensors can be concatenated");
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
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void ConcatLayer<T>::fprop(bool is_train) {
  prop_common(true, get_in_tensors(is_train), get_gpu().get_stream(), get_gpu().get_sm_count());
}

template <typename T>
void ConcatLayer<T>::bprop() {
  prop_common(false, get_in_tensors(true), get_gpu().get_stream(), get_gpu().get_sm_count());
}

template <typename T>
void ConcatLayer<T>::prop_common(bool forward, Tensors2<T>& in_tensors, cudaStream_t stream,
                                 size_t n_sms) {
  CudaDeviceContext context(get_device_id());

  int n_in_tensors = in_tensors.size();
  if (n_in_tensors == 2) {
    std::vector<InParam> in_params = set_in_params(in_tensors, 2);
    kernel_launch(forward, stream, n_sms, in_params[0], in_params[1]);
  } else if (n_in_tensors == 3) {
    std::vector<InParam> in_params = set_in_params(in_tensors, 3);
    kernel_launch(forward, stream, n_sms, in_params[0], in_params[1], in_params[2]);
  } else if (n_in_tensors == 4) {
    std::vector<InParam> in_params = set_in_params(in_tensors, 4);
    kernel_launch(forward, stream, n_sms, in_params[0], in_params[1], in_params[2], in_params[3]);
  } else if (n_in_tensors == 5) {
    std::vector<InParam> in_params = set_in_params(in_tensors, 5);
    kernel_launch(forward, stream, n_sms, in_params[0], in_params[1], in_params[2], in_params[3],
                  in_params[4]);
  } else {
    CK_THROW_(Error_t::UnSupportedFormat, "Merging > 5 layers is not supported");
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
std::vector<typename ConcatLayer<T>::InParam> ConcatLayer<T>::set_in_params(Tensors2<T>& in_tensors,
                                                                            int n) {
  std::vector<InParam> in_params;
  for (int i = 0; i < n; i++) {
    Tensor2<T>& in_tensor = in_tensors[i];
    T* in = in_tensor.get_ptr();
    int w = in_tensor.get_dimensions()[1];
    in_params.push_back({in, w});
  }
  return in_params;
}

template <typename T>
template <typename... Args>
void ConcatLayer<T>::kernel_launch(bool forward, cudaStream_t stream, size_t n_sms, Args&... args) {
  int block_size = 256;
  int n_blocks = n_sms * 8;
  Tensor2<T>& out_tensor = out_tensor_;
  T* out = out_tensor.get_ptr();
  int h = out_tensor.get_dimensions()[0];
  int out_w = out_tensor.get_dimensions()[1];
  concat_kernel<<<n_blocks, block_size, 0, stream>>>(forward, out, h, out_w, args...);
}

template <typename T>
Tensors2<T>& ConcatLayer<T>::get_in_tensors(bool is_train) {
  return in_tensors_;
}

template class ConcatLayer<float>;
template class ConcatLayer<__half>;

}  // namespace HugeCTR
