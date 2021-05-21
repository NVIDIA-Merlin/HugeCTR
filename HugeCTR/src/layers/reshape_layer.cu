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
#include <layers/reshape_layer.hpp>
#include <utils.hpp>

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

template <typename T>
ReshapeLayer<T>::ReshapeLayer(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                              size_t leading_dim, const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource),
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
ReshapeLayer<T>::ReshapeLayer(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                              const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                              std::vector<int>& selected,
                              const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource),
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
void ReshapeLayer<T>::initialize() {
  if (!in_place_) {
    CK_CUDA_THROW_(cudaMemcpyAsync(selected_tensor_.get_ptr(), &selected_.front(),
                                   n_active_slot_ * sizeof(int), cudaMemcpyHostToDevice,
                                   get_gpu().get_stream()));
  }
}

template <typename T>
void ReshapeLayer<T>::fprop(bool is_train) {
  prop_common(true, is_train, get_gpu().get_stream());
}

template <typename T>
void ReshapeLayer<T>::bprop() {
  prop_common(false, true, get_gpu().get_stream());
}

template <typename T>
void ReshapeLayer<T>::prop_common(bool forward, bool is_train, cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  Tensor2<T>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<T>& out_tensor = out_tensors_[0];

  if (in_place_) {
    if (forward) {
      cudaMemcpyAsync(out_tensor.get_ptr(), in_tensor.get_ptr(), in_tensor.get_size_in_bytes(),
                      cudaMemcpyDeviceToDevice, stream);
    } else {
      cudaMemcpyAsync(in_tensor.get_ptr(), out_tensor.get_ptr(), out_tensor.get_size_in_bytes(),
                      cudaMemcpyDeviceToDevice, stream);
    }
  } else {
    int block_size = 128;
    int n_block = get_gpu().get_sm_count() * 16;
    T* in = in_tensor.get_ptr();
    T* out = out_tensor.get_ptr();
    reshape_kernel<<<n_block, block_size>>>(in, out, batch_size_, n_slot_, vector_length_,
                                            selected_tensor_.get_ptr(), n_active_slot_, forward);
  }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename T>
Tensors2<T>& ReshapeLayer<T>::get_in_tensors(bool is_train) {
  return in_tensors_;
}

template class ReshapeLayer<float>;
template class ReshapeLayer<__half>;

}  // namespace HugeCTR
