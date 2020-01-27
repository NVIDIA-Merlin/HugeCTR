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

#include "HugeCTR/include/layers/reshape_layer.hpp"

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
__global__ void reshape_kernel(T* input, T* output, int batch_size, int n_slot, int vector_length,
                              const int* const selected, int n_active_slot, bool forward) {
  int base = blockIdx.x * blockDim.x + threadIdx.x;
  int input_vector_length = n_slot * vector_length;
  int output_vector_length = n_active_slot * vector_length;
  int output_n_elem = batch_size * output_vector_length;
  for (int idx = base; idx < output_n_elem; idx += blockDim.x * gridDim.x) {
    int batch_id = idx / output_vector_length;
    int idx_in_batch = idx % output_vector_length;
    int output_slot_id = idx_in_batch / vector_length;
    int idx_in_slot = idx_in_batch % vector_length;
    int input_slot_id = __ldg(&selected[output_slot_id]);
    int input_idx = batch_id * input_vector_length + input_slot_id * vector_length + idx_in_slot;
    if (forward)
      output[idx] = input[input_idx];
    else
      input[input_idx] = output[idx];
  }
}

}  // anonymous namespace


ReshapeLayer::ReshapeLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
                           std::shared_ptr<Tensor<float>>& out_tensor,
                           int leading_dim,
                           int device_id)
    : Layer(device_id),
      in_place_(true),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(0),
      selected_(nullptr),
      n_sms_(0) {

  try {
    CudaDeviceContext context(device_id);

    std::vector<int> in_dims = in_tensor->get_dims();
    int im_idx = in_dims.size() - 1;
    if(leading_dim < in_dims[im_idx] || leading_dim % in_dims[im_idx] != 0) {
        CK_THROW_(Error_t::WrongInput,
            "leading_dim < in_dims[im_idx] or leading_dim % in_dims[2] != 0");
    }

    int n_in_elems = in_tensor->get_num_elements();
    if(leading_dim > n_in_elems) {
        CK_THROW_(Error_t::WrongInput,
            "leading_dim cannot be bigger than n_in_elems");
    }

    if(n_in_elems % leading_dim != 0) {
        CK_THROW_(Error_t::WrongInput,
            "n_in_elems % leading_dim != 0");
    }

    int trailing_dim = n_in_elems / leading_dim;
    std::vector<int> out_dims = {trailing_dim, leading_dim};
    out_tensor.reset(new Tensor<float>(out_dims, *in_tensor, TensorFormat_t::HW));

    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

ReshapeLayer::ReshapeLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
                           std::shared_ptr<Tensor<float>>& out_tensor,
                           const std::shared_ptr<GeneralBuffer<float>>& blobs_buff,
                           std::vector<int>& selected,
                           int device_id)
    : Layer(device_id),
      in_place_(selected.empty()),
      batch_size_(0),
      n_slot_(0),
      vector_length_(0),
      n_active_slot_(selected.size()),
      selected_(nullptr),
      n_sms_(0) {
  try {
    CudaDeviceContext context(device_id);

    if (in_tensor->get_format() != TensorFormat_t::HSW) {
      CK_THROW_(Error_t::WrongInput, "Input format is invalid");
    }

    std::vector<int> in_dims = in_tensor->get_dims();
    if(in_dims[1] < n_active_slot_) {
      CK_THROW_(Error_t::WrongInput, "selected is invalid");
    }

    int in_dims_1 = selected.empty() ? in_dims[1] : int(n_active_slot_);
    std::vector<int> out_dims = {in_dims[0], in_dims_1 * in_dims[2]};

    if(in_place_) {
      out_tensor.reset(new Tensor<float>(out_dims, *in_tensor, TensorFormat_t::HW));
    }
    else {
      out_tensor.reset(new Tensor<float>(out_dims, blobs_buff, TensorFormat_t::HW));
      unsigned int i = 0;
      for (; i < in_dims.size() - 2; i++) batch_size_ += in_dims[i];
      n_slot_ = in_dims[i++];
      vector_length_ = in_dims[i];
      CK_CUDA_THROW_(cudaMalloc(&selected_, n_active_slot_ * sizeof(int)));
      CK_CUDA_THROW_(cudaMemcpy(selected_, &selected.front(), n_active_slot_ * sizeof(int),
                                cudaMemcpyHostToDevice));
    }
    in_tensors_.emplace_back(in_tensor);
    out_tensors_.emplace_back(out_tensor);

    int device = get_device_id();
    CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sms_, cudaDevAttrMultiProcessorCount, device));
    assert(n_sms_ > 0);

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}


ReshapeLayer::~ReshapeLayer() {
  if (selected_) {
    cudaFree(selected_);
  }
}

void ReshapeLayer::fprop(cudaStream_t stream) {
  prop_common(true, stream);
}

void ReshapeLayer::bprop(cudaStream_t stream) {
  prop_common(false, stream);
}

void ReshapeLayer::prop_common(bool forward, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  if (!in_place_) {
    int block_size = 128;
    int n_block = n_sms_ * 16;
    const auto& in_tensor = in_tensors_[0];
    const auto&  out_tensor = out_tensors_[0];
    float* in = in_tensor->get_ptr();
    float* out = out_tensor->get_ptr();
    reshape_kernel<<<n_block, block_size>>>(in, out, batch_size_, n_slot_, vector_length_, selected_,
                                           n_active_slot_, forward);
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}


}  // namespace HugeCTR
