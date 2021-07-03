

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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {

namespace {

// fuse backward_reorder_kernel + all2all + backward_sum_kernel into one kernel
template <typename TypeEmbeddingComp>
__global__ void backward_sum_fuse_kernel(size_t local_gpu_id, size_t gpu_num, size_t batch_size,
                                         size_t batch_size_per_gpu, size_t slot_num,
                                         size_t slot_num_per_gpu, size_t embedding_vec_size,
                                         TypeEmbeddingComp *const *embedding_features,
                                         TypeEmbeddingComp *wgrad) {
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  // for(int bid = blockIdx.x; bid < batch_size; bid += gridDim.x) {
  int offset = (local_gpu_id + 1) * batch_size_per_gpu;
  for (int bid_offset = (blockIdx.x + offset); bid_offset < (batch_size + offset);
       bid_offset += gridDim.x) {
    int bid = bid_offset % batch_size;

    if (tid < embedding_vec_size) {
      int gpu_id = bid / batch_size_per_gpu;     // source gpu id
      int sample_id = bid % batch_size_per_gpu;  // sample id on source gpu

      for (int i = 0; i < slot_num_per_gpu; i++) {
        int slot_id =
            i * gpu_num + local_gpu_id;  // slot id on source gpu (for localizedSlotEmbedding)
        size_t feature_id = sample_id * slot_num + slot_id;  // feature id on source gpu

        // load dgrad from multi-gpu and store the wgrad in local gpu
        size_t src_feature_index = feature_id * embedding_vec_size + tid;
        size_t dst_feature_index = (size_t)(bid * slot_num_per_gpu + i) * embedding_vec_size + tid;
        wgrad[dst_feature_index] = embedding_features[gpu_id][src_feature_index];
      }
    }
  }
}

// overload for fp16
__global__ void backward_sum_fuse_align2_kernel(size_t local_gpu_id, size_t gpu_num,
                                                size_t batch_size, size_t batch_size_per_gpu,
                                                size_t slot_num, size_t slot_num_per_gpu,
                                                size_t embedding_vec_size,
                                                __half *const *embedding_features, __half *wgrad) {
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  // for(int bid = blockIdx.x; bid < batch_size; bid += gridDim.x) {
  int offset = (local_gpu_id + 1) * batch_size_per_gpu;
  for (int bid_offset = (blockIdx.x + offset); bid_offset < (batch_size + offset);
       bid_offset += gridDim.x) {
    int bid = bid_offset % batch_size;

    if (tid < embedding_vec_size) {
      __half2 *const *embedding_features2 = reinterpret_cast<__half2 *const *>(embedding_features);
      __half2 *wgrad2 = reinterpret_cast<__half2 *>(wgrad);

      int gpu_id = bid / batch_size_per_gpu;     // target gpu id
      int sample_id = bid % batch_size_per_gpu;  // sample id on target gpu

      for (int i = 0; i < slot_num_per_gpu; i++) {
        int slot_id =
            i * gpu_num + local_gpu_id;  // slot id on source gpu (for localizedSlotEmbedding)
        size_t feature_id = sample_id * slot_num + slot_id;  // feature id on source gpu

        // load dgrad from multi-gpu and store the wgrad in local gpu
        size_t src_feature_index = feature_id * embedding_vec_size + tid;
        size_t dst_feature_index = (size_t)(bid * slot_num_per_gpu + i) * embedding_vec_size + tid;
        wgrad2[dst_feature_index] = embedding_features2[gpu_id][src_feature_index];
      }
    }
  }
}

template <typename TypeEmbeddingComp>
void backward_fuse(size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
                   size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size,
                   TypeEmbeddingComp *const *embedding_features, TypeEmbeddingComp *wgrad,
                   size_t sm_count, cudaStream_t stream) {
  const size_t block_size = embedding_vec_size;
  int maxActiveBlocks;
  CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, backward_sum_fuse_kernel<TypeEmbeddingComp>, block_size, 0));
  const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

  backward_sum_fuse_kernel<<<grid_size, block_size, 0, stream>>>(
      id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
      embedding_vec_size, embedding_features, wgrad);
}

void backward_fuse(size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
                   size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size,
                   __half *const *embedding_features, __half *wgrad, size_t sm_count,
                   cudaStream_t stream) {
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    int maxActiveBlocks;
    CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, backward_sum_fuse_align2_kernel, block_size, 0));
    const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

    backward_sum_fuse_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
        embedding_vec_size / 2, embedding_features, wgrad);
  } else {
    const size_t block_size = embedding_vec_size;
    int maxActiveBlocks;
    CK_CUDA_THROW_(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, backward_sum_fuse_kernel<__half>, block_size, 0));
    const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

    backward_sum_fuse_kernel<<<grid_size, block_size, 0, stream>>>(
        id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
        embedding_vec_size, embedding_features, wgrad);
  }
}

}  // namespace

/**
 * backward propagation for LocalizedSlotSparseEmbeddingOneHot (per gpu).
 * fuse (backward_reorder + all2all + backward_xxx_kernel) into one kernel.
 * Only support single node currently.
 * @param id local gpu id
 * @param local_gpu_count local gpu count
 * @param batch_size batch size for the current mini-batch computation
 * @param batch_size_per_gpu batchsize per gpu
 * @param slot_num total slots number
 * @param slot_num_per_gpu the number of slots for each GPU
 * @param embedding_vec_size embedding vector size.
 * @param combiner 0-sum; 1-mean
 * @param embedding_features embedding features of all gpus (output)
 * @param wgrad wgrad, the output of this function.
 * @param stream cuda stream
 */
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward_fuse_per_gpu(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<TypeEmbeddingComp *> &embedding_features, Tensor2<TypeEmbeddingComp> &wgrad,
    size_t sm_count, cudaStream_t stream) {
  if (combiner == 0) {
    backward_fuse(id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
                  embedding_vec_size, embedding_features.get_ptr(), wgrad.get_ptr(), sm_count,
                  stream);
  } else {
    CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
  }

  return;
}

template void SparseEmbeddingFunctors::backward_fuse_per_gpu<float>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<float *> &embedding_features, Tensor2<float> &wgrad, size_t sm_count,
    cudaStream_t stream);

template void SparseEmbeddingFunctors::backward_fuse_per_gpu<__half>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<__half *> &embedding_features, Tensor2<__half> &wgrad, size_t sm_count,
    cudaStream_t stream);

}  // namespace HugeCTR