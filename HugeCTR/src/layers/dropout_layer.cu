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

#include "HugeCTR/include/layers/dropout_layer.hpp"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <functional>
#include "HugeCTR/include/utils.hpp"
#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

__global__ void dropout_kernel(const float* in, const float* mask, float* out,
                               const int len, const float rate) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    out[i] = (mask[i] > rate) * in[i];
  }
}


} // end namespace

DropoutLayer::DropoutLayer(const std::shared_ptr<Tensor<float>>& in_tensor,
                           const std::shared_ptr<Tensor<float>>& out_tensor,
                           float rate,
                           const curandGenerator_t& curand_generator,
                           int device_id)
    : Layer(device_id),
      rate_(rate),
      mask_(nullptr),
      curand_generator_(curand_generator),
      n_sms_(0) {
  assert(get_size_from_dims(in_tensor->get_dims()) == get_size_from_dims(out_tensor->get_dims()));
  assert(rate_ > 0.f && rate_ < 1.f);

  in_tensors_.emplace_back(in_tensor);
  out_tensors_.emplace_back(out_tensor);

  CudaDeviceContext context(get_device_id());
  CK_CUDA_THROW_(cudaMalloc(&mask_, in_tensor->get_size()));
  CK_CURAND_THROW_(curandSetPseudoRandomGeneratorSeed(curand_generator_, get_seed()));

  CK_CUDA_THROW_(cudaDeviceGetAttribute(&n_sms_, cudaDevAttrMultiProcessorCount, get_device_id()));
  assert(n_sms_ > 0);
}

DropoutLayer::~DropoutLayer() {
  if (mask_) {
    cudaFree(mask_);
  }
}

void DropoutLayer::fprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  CK_CURAND_THROW_(curandGenerateUniform(curand_generator_, mask_, in_tensors_[0]->get_num_elements()));
  prop_common(in_tensors_[0]->get_ptr(), out_tensors_[0]->get_ptr(), stream);
}

void DropoutLayer::bprop(cudaStream_t stream) {
  CudaDeviceContext context(get_device_id());
  prop_common(out_tensors_[0]->get_ptr(), in_tensors_[0]->get_ptr(), stream);
}

int64_t DropoutLayer::get_seed() const {
  FILE* f = fopen("/dev/urandom", "rb");
  if(f) {
    int64_t seed;
    size_t ret = fread(&seed, 1, sizeof(seed), f);
    fclose(f);
    if(ret == sizeof(seed)) {
      return seed;
    }
  }
  return time(nullptr);
}

void DropoutLayer::prop_common(const float* in, float* out, cudaStream_t stream) {
  int len = in_tensors_[0]->get_num_elements();

  int grid_size = n_sms_ * 16;
  int block_size = 256;
  dropout_kernel<<<grid_size, block_size>>>(in, mask_, out, len, rate_);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

}  // namespace HugeCTR
