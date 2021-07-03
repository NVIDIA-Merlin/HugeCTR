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

#include <cuda_fp16.h>

#include <layers/cast_layer.hpp>

#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace {

__global__ void cast_kernel(__half* out, const float* in, int size) {
  __half2* out2 = (__half2*)(out);
  float2* in2 = (float2*)(in);
  int size2 = size / 2;

  int start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = start; i < size2; i += blockDim.x * gridDim.x) {
    out2[i] = __float22half2_rn(__ldg(in2 + i));
  }
  if (start == 0 && size % 2 > 0) {
    out[size - 1] = __float2half(__ldg(in + size - 1));
  }
}

__global__ void cast_kernel(float* out, const __half* in, int size) {
  float2* out2 = (float2*)(out);
  __half2* in2 = (__half2*)(in);
  int size2 = size / 2;

  int start = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = start; i < size2; i += blockDim.x * gridDim.x) {
    out2[i] = __half22float2(__ldg(in2 + i));
  }
  if (start == 0 && size % 2 > 0) {
    out[size - 1] = __half2float(__ldg(in + size - 1));
  }
}

}  // namespace

template <typename From, typename To>
CastLayer<From, To>::CastLayer(const Tensor2<From>& bottom_tensor, const Tensor2<To>& top_tensor,
                               const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  assert(bottom_tensor.get_num_elements() == top_tensor.get_num_elements());

  bottom_tensor_ = bottom_tensor;
  top_tensor_ = top_tensor;
}

template <typename From, typename To>
void CastLayer<From, To>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const From* bottom = bottom_tensor_.get_ptr();
  To* top = top_tensor_.get_ptr();

  const size_t threads = 512;
  const size_t blocks = std::min((bottom_tensor_.get_num_elements() - 1) / threads + 1, 1024ul);
  cast_kernel<<<blocks, threads, 0, get_gpu().get_stream()>>>(top, bottom,
                                                              bottom_tensor_.get_num_elements());

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template <typename From, typename To>
void CastLayer<From, To>::bprop() {
  CudaDeviceContext context(get_device_id());

#ifndef NDEBUG
  CK_CUDA_THROW_(cudaDeviceSynchronize());
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

template class CastLayer<float, __half>;
template class CastLayer<__half, float>;

}  // namespace HugeCTR
