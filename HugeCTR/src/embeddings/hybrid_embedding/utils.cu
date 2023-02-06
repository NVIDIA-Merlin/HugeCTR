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

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <common.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <iostream>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, const Tensor2<dtype> tensor,
                     cudaStream_t stream) {
  size_t tensor_size = tensor.get_num_elements();
  h_tensor.resize(tensor_size);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  HCTR_LIB_THROW(cudaMemcpyAsync(h_tensor.data(), tensor.get_ptr(), tensor.get_size_in_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

template <typename dtype>
void upload_tensor(const std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream) {
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  assert(tensor.get_num_elements() >= h_tensor.size());
  HCTR_LIB_THROW(cudaMemcpyAsync(tensor.get_ptr(), h_tensor.data(), h_tensor.size() * sizeof(dtype),
                                 cudaMemcpyHostToDevice, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

__global__ void offsets_kernel(const uint32_t* indices, uint32_t* indices_offsets,
                               uint32_t num_instances, uint32_t multiplier) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < num_instances) {
    uint32_t searched_value = multiplier * tid;
    uint32_t num_selected = indices_offsets[num_instances];

    // Binary search
    uint32_t i = 0;
    uint32_t j = num_selected;
    while (i < j) {
      uint32_t m = (i + j) / 2;
      uint32_t value = __ldg(indices + m);

      if (value < searched_value)
        i = m + 1;
      else
        j = m;
    }

    // Write offset
    indices_offsets[tid] = i;
  }
}

template <typename dtype, typename stype>
__global__ void modulo_kernel(dtype* buffer, const stype* d_num_elements, dtype divisor) {
  const stype num_elements = __ldg(d_num_elements);
  for (stype i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += blockDim.x * gridDim.x)
    buffer[i] %= divisor;
}

__global__ void model_id_kernel(const uint32_t* indices_offsets, uint32_t* src_model_id,
                                const uint32_t* d_num_elements) {
  // Find model id
  uint32_t num_elements = __ldg(d_num_elements);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += blockDim.x * gridDim.x) {
    uint32_t model_id = 0;
    uint32_t next_offset = indices_offsets[1];
    while (next_offset <= i) {
      model_id++;
      next_offset = indices_offsets[model_id + 1];
    }
    src_model_id[i] = model_id;
  }
}

template void download_tensor<uint32_t>(std::vector<uint32_t>& h_tensor,
                                        const Tensor2<uint32_t> tensor, cudaStream_t stream);
template void download_tensor<unsigned long>(std::vector<size_t>& h_tensor,
                                             const Tensor2<size_t> tensor, cudaStream_t stream);
template void download_tensor<long long>(std::vector<long long>& h_tensor,
                                         const Tensor2<long long> tensor, cudaStream_t stream);
template void download_tensor<__half>(std::vector<__half>& h_tensor, const Tensor2<__half> tensor,
                                      cudaStream_t stream);
template void download_tensor<float>(std::vector<float>& h_tensor, const Tensor2<float> tensor,
                                     cudaStream_t stream);
template void upload_tensor<uint32_t>(const std::vector<uint32_t>& h_tensor,
                                      Tensor2<uint32_t> tensor, cudaStream_t stream);
template void upload_tensor<unsigned long>(const std::vector<size_t>& h_tensor,
                                           Tensor2<size_t> tensor, cudaStream_t stream);
template void upload_tensor<long long>(const std::vector<long long>& h_tensor,
                                       Tensor2<long long> tensor, cudaStream_t stream);

template void upload_tensor<__half>(const std::vector<__half>& h_tensor, Tensor2<__half> tensor,
                                    cudaStream_t stream);
template void upload_tensor<float>(const std::vector<float>& h_tensor, Tensor2<float> tensor,
                                   cudaStream_t stream);

template __global__ void modulo_kernel(uint32_t* buffer, const uint32_t* d_num_elements,
                                       uint32_t divisor);
}  // namespace hybrid_embedding

}  // namespace HugeCTR