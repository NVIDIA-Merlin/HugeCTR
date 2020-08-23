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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {

namespace {

// store slot_id by row_offset and value_index
template <typename TypeKey, typename TypeValueIndex>
__global__ void store_slot_id_kernel(size_t batch_size,
                                     int slot_num,  // total slot number in hash table
                                     int slot_num_per_gpu,
                                     int gpu_num,  // total gpu number
                                     int gpu_id,   // global gpu device id
                                     const TypeKey *row_offset, const TypeValueIndex *value_index,
                                     TypeValueIndex *slot_id) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < (batch_size * slot_num_per_gpu)) {
    int sid = gid % slot_num_per_gpu;
    sid = gpu_id + sid * gpu_num;  // global slot id
    if (sid < slot_num) {
      TypeKey offset = row_offset[gid];
      int value_num = row_offset[gid + 1] - offset;
      for (int i = 0; i < value_num; i++) {
        TypeValueIndex index = value_index[offset + i];  // row number
        // TODO: slot_id may be filled in repeatly
        slot_id[index] = sid;
      }
    }
  }
}

}  // namespace

template <typename TypeKey>
void SparseEmbeddingFunctors::store_slot_id(size_t batch_size, size_t slot_num,
                                            const std::vector<size_t> &slot_num_per_gpu,
                                            const Tensors2<TypeKey> &row_offset_tensors,
                                            const Tensors2<size_t> &value_index_tensors,
                                            Tensors2<size_t> &slot_id_tensors,
                                            const GPUResourceGroup &device_resources) {
  CudaDeviceContext context;
  size_t local_gpu_count = device_resources.size();
  size_t total_gpu_count = device_resources.get_total_gpu_count();

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (slot_num_per_gpu[id] == 0) {
      continue;
    }

    size_t local_device_id = device_resources[id].get_device_id();
    size_t global_id = device_resources.get_global_id(local_device_id);

    const size_t block_size = 64;
    const size_t grid_size = (batch_size * slot_num_per_gpu[id] + block_size - 1) / block_size;

    context.set_device(local_device_id);
    store_slot_id_kernel<<<grid_size, block_size, 0, device_resources[id].get_stream()>>>(
        batch_size, slot_num, slot_num_per_gpu[id], total_gpu_count, global_id,
        row_offset_tensors[id].get_ptr(), value_index_tensors[id].get_ptr(),
        slot_id_tensors[id].get_ptr());
  }
}

template void SparseEmbeddingFunctors::store_slot_id<unsigned int>(
    size_t batch_size, size_t slot_num, const std::vector<size_t> &slot_num_per_gpu,
    const Tensors2<unsigned int> &row_offset_tensors, const Tensors2<size_t> &value_index_tensors,
    Tensors2<size_t> &slot_id_tensors, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::store_slot_id<long long>(
    size_t batch_size, size_t slot_num, const std::vector<size_t> &slot_num_per_gpu,
    const Tensors2<long long> &row_offset_tensors, const Tensors2<size_t> &value_index_tensors,
    Tensors2<size_t> &slot_id_tensors, const GPUResourceGroup &device_resources);

}  // namespace HugeCTR