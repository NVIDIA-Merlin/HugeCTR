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
#include <layers/gather_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

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
GatherLayer<T>::GatherLayer(const core23::Tensor& input_tensor, core23::Tensor& output_tensor,
                            std::vector<int>& indices,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource), h_indices_(indices) {
  try {
    if (indices.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty slice indices is not allowed");
    }
    // input tensor is 2D.
    // dim_0 represents the most outside dimension.
    // dim_1 represents the multiplication of the rest dimensions.
    tensor_size_ = (size_t)input_tensor.shape().size(1);
    size_t tensor_num = (size_t)input_tensor.shape().size(0);

    num_indices_ = indices.size();

    for (size_t i = 0; i < num_indices_; i++) {
      if (indices.data()[i] > int(tensor_num) - 1)
        HCTR_OWN_THROW(Error_t::WrongInput, "Index is out of range");
    }
    indices23_ = core23::Tensor(core23::Shape({(int64_t)num_indices_}),
                                core23::DataType(core23::ScalarType::Int32));
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename T>
void GatherLayer<T>::initialize() {
  HCTR_LIB_THROW(cudaMemcpyAsync(indices23_.data(), (void*)h_indices_.data(),
                                 num_indices_ * sizeof(int), cudaMemcpyHostToDevice,
                                 get_gpu().get_stream()));
}

template <typename T>
void GatherLayer<T>::fprop(bool is_train) {
  int block_size = 512;
  int n_blocks = get_gpu().get_sm_count() * 4;
  core23::Tensor& in_tensor = input_tensors_[0];
  core23::Tensor& out_tensor = output_tensors_[0];
  T* out = out_tensor.data<T>();
  T* in = in_tensor.data<T>();
  gather_kernel<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(
      true, in, out, tensor_size_, num_indices_, static_cast<int*>(indices23_.data()));
}

template <typename T>
void GatherLayer<T>::bprop() {
  int block_size = 512;
  int n_blocks = get_gpu().get_sm_count() * 4;
  core23::Tensor& in_tensor = input_tensors_[0];
  core23::Tensor& out_tensor = output_tensors_[0];
  T* out = out_tensor.data<T>();
  T* in = in_tensor.data<T>();
  int h = in_tensor.shape().size(0);
  initialize_array<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(in, h * tensor_size_, T(0));
  gather_kernel<<<n_blocks, block_size, 0, get_gpu().get_stream()>>>(
      false, in, out, tensor_size_, num_indices_, static_cast<int*>(indices23_.data()));
}

template class GatherLayer<float>;

}  // namespace HugeCTR
