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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/sequence_mask_layer.hpp>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

namespace {

template <typename T>
__global__ void build_sequence_mask_kernel(T* attention_mask, const T* sequence_lengths,
                                           int max_seq_len) {
  // sequence_lengths: [batch_size]
  // attention_mask: [batch_size, 1, 1, max_seq_len]
  attention_mask += blockIdx.x * max_seq_len;
  const int length = sequence_lengths[blockIdx.x];
  for (int i = threadIdx.x; i < max_seq_len; i += blockDim.x) {
    if (i < length) {
      attention_mask[i] = (T)(1.0f);
    } else {
      attention_mask[i] = (T)(0.0f);
    }
  }
}

}  // namespace

template <typename T>
SequenceMaskLayer<T>::SequenceMaskLayer(
    const Tensor2<T>& in_tensor, const Tensor2<T>& out_tensor, int max_sequence_len,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource), max_sequence_len_(max_sequence_len) {
  assert(in_tensor.get_dimensions() == 2);

  in_tensor_ = in_tensor;
  out_tensor_ = out_tensor;
}

template <typename T>
void SequenceMaskLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  int max_sequence_len = max_sequence_len_;

  T* input = in_tensor_.get_ptr();
  T* output = out_tensor_.get_ptr();

  const size_t batch_size = in_tensor_.get_dimensions()[0];
  const size_t block_dim = 512;

  build_sequence_mask_kernel<<<batch_size, block_dim, 0, get_gpu().get_stream()>>>(
      output, input, max_sequence_len);

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template <typename T>
void SequenceMaskLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

#ifndef NDEBUG
  cudaDeviceSynchronize();
  HCTR_LIB_THROW(cudaGetLastError());
#endif
}

template class SequenceMaskLayer<float>;
template class SequenceMaskLayer<__half>;

}  // namespace HugeCTR
