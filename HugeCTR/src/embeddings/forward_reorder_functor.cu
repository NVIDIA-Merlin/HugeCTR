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
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace {

// reorder operation after all2all in forward propagation
template <typename TypeEmbeddingComp>
__global__ void forward_reorder_kernel(int batch_size_per_gpu, int slot_num, int embedding_vec_size,
                                       int gpu_num, const TypeEmbeddingComp *input,
                                       TypeEmbeddingComp *output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding
  // vector gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one
  // sample on each GPU Each thread needs to process slot_num slots

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid;  // sample_id on the current GPU

  if ((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    int dst_offset =
        sample_id * slot_num * embedding_vec_size;  // offset for the first slot of one sample
    int dst_stride = embedding_vec_size;            // stride from slot to slot

    for (int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num;
      int offset_pre = 0;  // offset in previous gpus
      for (int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id < (slot_num % gpu_num)) ? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;
      }
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id < (slot_num % gpu_num)) ? 1 : 0);
      int offset_cur = sample_id * slot_num_per_gpu;  // offset in current gpu
      int src_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int dst_addr = dst_offset + dst_stride * slot_id;
      output[dst_addr + tid] = input[src_addr + tid];
    }
  }
}

// reorder operation after all2all in forward propagation
__global__ void forward_reorder_align2_kernel(int batch_size_per_gpu, int slot_num,
                                              int embedding_vec_size, int gpu_num,
                                              const __half *input, __half *output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding
  // vector gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one
  // sample on each GPU Each thread needs to process slot_num slots

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid;  // sample_id on the current GPU

  if ((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    const __half2 *input2 = reinterpret_cast<const __half2 *>(input);
    __half2 *output2 = reinterpret_cast<__half2 *>(output);

    int dst_offset =
        sample_id * slot_num * embedding_vec_size;  // offset for the first slot of one sample
    int dst_stride = embedding_vec_size;            // stride from slot to slot

    for (int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num;
      int offset_pre = 0;  // offset in previous gpus
      for (int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id < (slot_num % gpu_num)) ? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;
      }
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id < (slot_num % gpu_num)) ? 1 : 0);
      int offset_cur = sample_id * slot_num_per_gpu;  // offset in current gpu
      int src_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int dst_addr = dst_offset + dst_stride * slot_id;
      output2[dst_addr + tid] = input2[src_addr + tid];
    }
  }
}

template <typename TypeEmbeddingComp>
void do_forward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        size_t total_gpu_count, const TypeEmbeddingComp *input,
                        TypeEmbeddingComp *output, cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  const size_t block_size = embedding_vec_size;
  forward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
}

void do_forward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        size_t total_gpu_count, const __half *input, __half *output,
                        cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_reorder_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size / 2, total_gpu_count, input, output);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
  }
}

}  // namespace

/**
 * reoder the sequence of data after all2all operation in forward propagation
 * @param batch_size_per_gpu batch size per GPU
 * @param slot_num the number of localized slots
 * @param embedding_vec_size embedding vector size.
 * @param src_tensors the source tensors before reorder
 * @param dst_tensors the destination tensors after reorder
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_reorder(size_t batch_size_per_gpu, size_t slot_num,
                                              size_t embedding_vec_size,
                                              const Tensors2<TypeEmbeddingComp> &src_tensors,
                                              Tensors2<TypeEmbeddingComp> &dst_tensors,
                                              const ResourceManager &resource_manager) {
  CudaDeviceContext context;
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());

    do_forward_reorder(batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
                       src_tensors[id].get_ptr(), dst_tensors[id].get_ptr(),
                       local_gpu->get_stream());
  }
}

template void SparseEmbeddingFunctors::forward_reorder<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<float> &src_tensors, Tensors2<float> &dst_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::forward_reorder<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<__half> &src_tensors, Tensors2<__half> &dst_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR