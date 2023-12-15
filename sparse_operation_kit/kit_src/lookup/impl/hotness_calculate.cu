/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "common/check.h"
#include "lookup/impl/hotness_calculate.h"

namespace sok {

template <typename DType>
__global__ void hotnessCalKernel(const DType *row_length_recv_buffer, size_t local_batchsize,
                                 int num_lookup, int num_gpus, int *outputs) {
  size_t thread_cnt = blockDim.x * gridDim.x;
  size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t items = local_batchsize * num_lookup * num_gpus;
  extern __shared__ int smem[];
  for (size_t i = threadIdx.x; i < num_lookup; i += blockDim.x) {
    smem[i] = 0;
  }

  __syncthreads();
  for (size_t i = thread_idx; i < items; i += thread_cnt) {
    size_t num_lookup_id = (i / local_batchsize) % num_lookup;
    int value = (int)(row_length_recv_buffer[i]);
    atomicMax(smem + num_lookup_id, value);
  }

  __syncthreads();
  for (size_t i = threadIdx.x; i < num_lookup; i += blockDim.x) {
    atomicMax(outputs + i, smem[i]);
  }
}

template <typename DType>
void HotnessCalLauncher<DType>::initialize() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename DType>
void HotnessCalLauncher<DType>::operator()(const void *row_length_recv_buffer,
                                           size_t local_batchsize, int num_lookup, int num_gpus,
                                           void *output_device, void *output_host,
                                           cudaStream_t stream) {
  const DType *t_row_length_recv_buffer = reinterpret_cast<const DType *>(row_length_recv_buffer);
  int32_t *t_output_device = reinterpret_cast<int32_t *>(output_device);
  int32_t *t_output_host = reinterpret_cast<int32_t *>(output_host);

  dim3 grid_dim(2 * sm_count_);
  dim3 block_dim(1024ul);
  CUDACHECK(cudaMemsetAsync(t_output_device, 0, sizeof(int32_t) * num_lookup, stream));
  hotnessCalKernel<DType><<<grid_dim, block_dim, num_lookup * sizeof(int32_t), stream>>>(
      t_row_length_recv_buffer, local_batchsize, num_lookup, num_gpus, t_output_device);
  CUDACHECK(cudaMemcpyAsync(t_output_host, t_output_device, sizeof(int32_t) * num_lookup,
                            cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  // CUDACHECK(cudaGetLastError());
}

template class HotnessCalLauncher<int32_t>;
template class HotnessCalLauncher<int64_t>;

}  // namespace sok
