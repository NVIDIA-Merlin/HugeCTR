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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/sequence_mask_layer.hpp>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void build_sequence_mask_kernel(T* attention_mask, const T* sequence_lengths_from,
                                           const T* sequence_lengths_to, int max_seq_len_from,
                                           int max_seq_len_to) {
  // sequence_lengths: [batch_size]
  // attention_mask: [batch_size, 1, max_seq_len_from, max_seq_len_to]
  attention_mask += blockIdx.x * max_seq_len_from * max_seq_len_to;
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int warp = threadIdx.x >> 5;    // warp idx
  const int length_from = sequence_lengths_from[blockIdx.x];
  const int length_to = sequence_lengths_to[blockIdx.x];
  for (int warp_id = warp; warp_id < max_seq_len_from; warp_id += ((blockDim.x) >> 5)) {
    for (int lane_id = lane; lane_id < max_seq_len_to; lane_id += 32) {
      // printf("BlockdIdx %d, ThreadIdx %d, laneId %d, warpId %d, length_from %d, length_to %d\n",
      //       blockIdx.x, threadIdx.x, lane_id, warp_id, length_from, length_to);
      if (warp_id < length_from && lane_id < length_to) {
        attention_mask[warp_id * max_seq_len_to + lane_id] = (T)(1.0f);
      } else {
        attention_mask[warp_id * max_seq_len_to + lane_id] = (T)(0.0f);
      }
    }
  }
}

}  // namespace

template <typename T>
SequenceMaskLayer<T>::SequenceMaskLayer(const std::vector<core23::Tensor>& input_tensors,
                                        const core23::Tensor& output_tensor,
                                        int max_sequence_len_from, int max_sequence_len_to,
                                        const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(input_tensors, {output_tensor}, gpu_resource),
      max_sequence_len_from_(max_sequence_len_from),
      max_sequence_len_to_(max_sequence_len_to) {
  assert(input_tensors_[0].shape().dims() == 2);
  assert(input_tensors_.size() == 2);
}

template <typename T>
void SequenceMaskLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  auto* seq_from_len = input_tensors_[0].data<T>();
  auto* seq_to_len = input_tensors_[1].data<T>();
  auto* output = output_tensors_[0].data<T>();

  const auto batch_size = input_tensors_[0].shape().size(0);
  const size_t block_dim = 1024;

  build_sequence_mask_kernel<<<batch_size, block_dim, 0, get_gpu().get_stream()>>>(
      output, seq_from_len, seq_to_len, max_sequence_len_from_, max_sequence_len_to_);
}

template <typename T>
void SequenceMaskLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  return;
}

template class SequenceMaskLayer<float>;
template class SequenceMaskLayer<__half>;

}  // namespace HugeCTR
