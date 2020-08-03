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

#include <layers/relu_layer_half.hpp>
#include <utils.cuh>

namespace HugeCTR {

namespace {

__global__ void forward_half2_relu_kernel(__half* top, const __half* bottom, int size) {
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* top2 = reinterpret_cast<__half2*>(top);
  const __half2* bottom2 = reinterpret_cast<const __half2*>(bottom);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    __half2 t = __ldg(bottom2 + i);
    __half2 mask = __hgt2(t, zero);
    top2[i] = __hmul2(t, mask);
  }
}

__global__ void backward_half2_relu_kernel(__half* bottom, const __half* top, int size) {
  const __half2 zero = TypeFunc<__half2>::zero();
  __half2* bottom2 = reinterpret_cast<__half2*>(bottom);
  const __half2* top2 = reinterpret_cast<const __half2*>(top);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    __half2 t = bottom2[i];
    half2 mask = __hgt2(t, zero);
    bottom2[i] = __hmul2(__ldg(top2 + i), mask);
  }
}

}  // namespace

ReluLayerHalf::ReluLayerHalf(const TensorPtr<__half>& bottom_tensor,
                             const TensorPtr<__half>& top_tensor, int device_id)
    : Layer(device_id) {
  assert(get_size_from_dims(bottom_tensor->get_dims()) ==
         get_size_from_dims(top_tensor->get_dims()));
  assert(get_size_from_dims(bottom_tensor->get_dims()) % 2 == 0);

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void ReluLayerHalf::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = get_size_from_dims(bottom_tensor_->get_dims()) / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  forward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(top_tensor_->get_ptr(),
                                                                bottom_tensor_->get_ptr(), size);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void ReluLayerHalf::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;
  const size_t MAX_GRID_DIM = 1024;

  const size_t size = get_size_from_dims(bottom_tensor_->get_dims()) / 2;
  const size_t grid_dim = std::min((size - 1) / BLOCK_DIM + 1, MAX_GRID_DIM);
  backward_half2_relu_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(bottom_tensor_->get_ptr(),
                                                                 top_tensor_->get_ptr(), size);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
