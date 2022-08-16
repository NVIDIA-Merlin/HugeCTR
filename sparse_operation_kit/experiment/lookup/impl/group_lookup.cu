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

#include <cuda_fp16.h>

#include "common/check.h"
#include "lookup/impl/group_lookup.h"

namespace sok {

template <typename KeyType, typename DataType>
__global__ static void FusedLookupKernel(LookupTask<KeyType, DataType> *task, size_t num_tasks) {
  for (size_t i = 0; i < num_tasks; ++i) {
    const float *input = task[i].input;
    const KeyType *key = reinterpret_cast<const KeyType *>(task[i].key);
    int32_t dimension = task[i].dimension;
    int32_t num_keys = task[i].num_keys;
    DataType *output = reinterpret_cast<DataType *>(task[i].output);

    size_t thread_cnt = blockDim.x * gridDim.x;
    size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t items = num_keys * dimension;
    for (size_t j = thread_idx; j < items; j += thread_cnt) {
      size_t row = key[j / dimension];
      size_t col = j % dimension;
      output[j] = input[row * dimension + col];
    }
  }
}

template <typename KeyType, typename DataType>
LookupLauncher<KeyType, DataType>::LookupLauncher() : num_tasks_(0), d_tasks_(nullptr) {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename KeyType, typename DataType>
LookupLauncher<KeyType, DataType>::~LookupLauncher() {
  if (d_tasks_) {
    CUDACHECK(cudaFree(d_tasks_));
    d_tasks_ = nullptr;
  }
}

template <typename KeyType, typename DataType>
void LookupLauncher<KeyType, DataType>::initialize(size_t num_tasks) {
  if (d_tasks_) return;
  num_tasks_ = num_tasks;
  CUDACHECK(cudaMalloc(&d_tasks_, sizeof(LookupTask<KeyType, DataType>) * num_tasks));
}

template <typename KeyType, typename DataType>
void LookupLauncher<KeyType, DataType>::operator()(
    std::vector<LookupTask<KeyType, DataType>> &h_tasks, cudaStream_t stream) {
  CUDACHECK(cudaMemcpyAsync(d_tasks_, h_tasks.data(),
                            sizeof(LookupTask<KeyType, DataType>) * h_tasks.size(),
                            cudaMemcpyHostToDevice, stream));
  FusedLookupKernel<KeyType, DataType><<<2 * sm_count_, 1024ul, 0, stream>>>(d_tasks_, num_tasks_);
  CUDACHECK(cudaGetLastError());
}

template class LookupLauncher<int64_t, float>;
template class LookupLauncher<int32_t, float>;
template class LookupLauncher<int64_t, __half>;
template class LookupLauncher<int32_t, __half>;

}  // namespace sok
