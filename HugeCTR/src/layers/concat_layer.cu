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

#include "HugeCTR/include/layers/concat_layer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
__global__ void concat_kernel(T* input, T* output, int n_batch, int n_slot, int vector_length,
                              const int* const slot_mask, int n_active_slot, bool forward) {
  int base = blockIdx.x * blockDim.x + threadIdx.x;
  int input_batch_size = n_slot * vector_length;
  int output_batch_size = n_active_slot * vector_length;
  int output_n_elem = n_batch * output_batch_size;
  for (int idx = base; idx < output_n_elem; idx += blockDim.x * gridDim.x) {
    int batch_id = idx / output_batch_size;
    int idx_in_batch = idx % output_batch_size;
    int output_slot_id = idx_in_batch / vector_length;
    int idx_in_slot = idx_in_batch % vector_length;
    int input_slot_id = __ldg(&slot_mask[output_slot_id]);
    int input_idx = batch_id * input_batch_size + input_slot_id * vector_length + idx_in_slot;
    if (forward)
      output[idx] = input[input_idx];
    else
      input[input_idx] = output[idx];
  }
}

}  // anonymous namespace

ConcatLayer::ConcatLayer(Tensor<float>& in_tensor, Tensor<float>& out_tensor,
                         std::vector<int>& slot_mask, int device_id)
    : Layer(device_id),
      in_place_(slot_mask.empty()),
      n_batch_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(slot_mask.size()),
      slot_mask_(nullptr),
      n_sm_(0) {
  try {
    CudaDeviceContext context(get_device_id());

    if (in_tensor.get_format() != TensorFormat_t::HSW ||
        out_tensor.get_format() != TensorFormat_t::HW)
      CK_THROW_(Error_t::WrongInput, "Input or output format is invalid");

    if (in_place_) {
      if (in_tensor.get_size() != out_tensor.get_size())
        CK_THROW_(Error_t::WrongInput, "Input and output tensors have inconsistent dims");
    } else {
      auto in_dims = in_tensor.get_dims();
      auto out_dims = out_tensor.get_dims();
      if (in_dims.size() != out_dims.size() + 1)
        CK_THROW_(Error_t::WrongInput, "Input and output tensors have inconsistent dims");
      for (unsigned int i = 0; i < out_dims.size(); i++) {
        if (i == out_dims.size() - 1) {
          if (in_dims[i + 1] * slot_mask.size() != (unsigned int)out_dims[i])
            CK_THROW_(Error_t::WrongInput, "The lowest dims of input/output is not compatible");
        } else {
          if (in_dims[i] != out_dims[i])
            CK_THROW_(Error_t::WrongInput, "The higher dims of input/output is mismatched");
        }
      }

      unsigned int i = 0;
      for (; i < in_dims.size() - 2; i++) n_batch_ += in_dims[i];
      n_slot_ = in_dims[i++];
      vector_length_ = in_dims[i];
      CK_CUDA_THROW_(cudaMalloc(&slot_mask_, n_active_slot_ * sizeof(int)));
      CK_CUDA_THROW_(cudaMemcpy(slot_mask_, &slot_mask.front(), n_active_slot_ * sizeof(int),
                                cudaMemcpyHostToDevice));
    }
    in_tensors_.push_back(std::ref(in_tensor));
    out_tensors_.push_back(std::ref(out_tensor));

    int device = get_device_id();
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sm_, cudaDevAttrMultiProcessorCount, device));
    assert(n_sm_ > 0);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

ConcatLayer::~ConcatLayer() {
  if (slot_mask_) cudaFree(slot_mask_);
}

void ConcatLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!in_place_) {
    int block_size = 128;
    int n_block = n_sm_ * 16;
    Tensor<float> in_tensor = in_tensors_[0];
    Tensor<float> out_tensor = out_tensors_[0];
    float* in = in_tensor.get_ptr();
    float* out = out_tensor.get_ptr();
    concat_kernel<<<n_block, block_size>>>(in, out, n_batch_, n_slot_, vector_length_, slot_mask_,
                                           n_active_slot_, true);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void ConcatLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  if (!in_place_) {
    int block_size = 128;
    int n_block = n_sm_ * 16;
    Tensor<float> in_tensor = in_tensors_[0];
    Tensor<float> out_tensor = out_tensors_[0];
    float* in = in_tensor.get_ptr();
    float* out = out_tensor.get_ptr();
    concat_kernel<<<n_block, block_size>>>(in, out, n_batch_, n_slot_, vector_length_, slot_mask_,
                                           n_active_slot_, false);
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
