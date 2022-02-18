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
#include <layers/gather_layer.hpp>
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

template <typename T>
__global__ void gather_kernel(bool forward, T* in, T* out, const int ts, const int n_idx,
                              int* indices) {
  for (int bidx = blockIdx.x; bidx < n_idx; bidx += gridDim.x) {
    int indx_begin = indices[bidx] * ts;
    int outdx_begin = bidx * ts;
    for (int i = threadIdx.x; i < ts; i += blockDim.x) {
      if (forward) {
        out[outdx_begin + i] = in[indx_begin + i];
      } else {
        in[indx_begin + i] = out[outdx_begin + i];
      }
    }
  }
  __syncthreads();
}

}  // anonymous namespace

template <typename T>
GatherLayer<T>::GatherLayer(const Tensor2<T>& in_tensor, Tensor2<T>& out_tensor,
                            const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                            std::vector<int>& indices,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), h_indices_(indices) {
  try {
    if (indices.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty slice indices is not allowed");
    }
    // input tensor is 2D.
    // dim_0 represents the most outside dimension.
    // dim_1 represents the multiplication of the rest dimensions.
    tensor_size = in_tensor.get_dimensions()[1];
    size_t tensor_num = in_tensor.get_dimensions()[0];

    num_indices = indices.size();

    for (size_t i = 0; i < num_indices; i++) {
      if (indices.data()[i] > int(tensor_num) - 1)
        HCTR_OWN_THROW(Error_t::WrongInput, "Index is out of range");
    }

    blobs_buff->reserve({num_indices}, &indices_);
    out_tensor_.push_back(out_tensor);
    in_tensors_.push_back(in_tensor);

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void GatherLayer<T>::initialize() {
  HCTR_LIB_THROW(cudaMemcpyAsync((void*)indices_.get_ptr(), (void*)h_indices_.data(),
                                 num_indices * sizeof(int), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void GatherLayer<T>::fprop(bool is_train) {
  int block_size = 512;
  int n_blocks = get_gpu().get_sm_count() * 4;
  Tensor2<T>& in_tensor = get_in_tensors(is_train)[0];
  Tensor2<T>& out_tensor = out_tensor_[0];
  T* out = out_tensor.get_ptr();
  T* in = in_tensor.get_ptr();
  gather_kernel<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(
      true, in, out, tensor_size, num_indices, indices_.get_ptr());
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void GatherLayer<T>::bprop() {
  int block_size = 512;
  int n_blocks = get_gpu().get_sm_count() * 4;
  Tensor2<T>& in_tensor = get_in_tensors(true)[0];
  Tensor2<T>& out_tensor = out_tensor_[0];
  T* out = out_tensor.get_ptr();
  T* in = in_tensor.get_ptr();
  int h = in_tensor.get_dimensions()[0];
  initialize_array<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(in, h * tensor_size, T(0));
  gather_kernel<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(
      false, in, out, tensor_size, num_indices, indices_.get_ptr());
#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class GatherLayer<float>;

}  // namespace HugeCTR
