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

#include <layers/cast_layer.hpp>

#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace {
__global__ void cast_kernel(__half* out, const float* in, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    out[i] = __float2half(__ldg(in + i));
  }
}

}  // namespace

CastLayer::CastLayer(const Tensor2<float>& bottom_tensor, const Tensor2<__half>& top_tensor,
                     const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(bottom_tensor.get_num_elements() == top_tensor.get_num_elements());

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

void CastLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const float* bottom = bottom_tensor_.get_ptr();
  __half* top = top_tensor_.get_ptr();

  const size_t threads = 512;
  const size_t blocks = std::min((bottom_tensor_.get_num_elements() - 1) / threads + 1, 1024ul);
  cast_kernel<<<blocks, threads, 0, get_gpu().get_stream()>>>(top, bottom,
                                                              bottom_tensor_.get_num_elements());

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void CastLayer::bprop() {
  CudaDeviceContext context(get_device_id());

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
