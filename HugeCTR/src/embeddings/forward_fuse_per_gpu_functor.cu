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

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.cuh>

namespace HugeCTR {

namespace {

// fuse foward_sum_kernel + all2all + forward_reorder into one kernel
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_sum_fuse_kernel(size_t local_gpu_id, size_t gpu_num, size_t batch_size,
                                        size_t batch_size_per_gpu, size_t slot_num,
                                        size_t slot_num_per_gpu, size_t embedding_vec_size,
                                        const TypeKey *row_offset, const size_t *hash_value_index,
                                        const float *hash_table_value,
                                        TypeEmbeddingComp **embedding_features) {
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  int offset = (local_gpu_id + 1) * batch_size_per_gpu;
  for (int bid_offset = (blockIdx.x + offset); bid_offset < (batch_size + offset);
       bid_offset += gridDim.x) {
    int bid = bid_offset % batch_size;

    if (tid < embedding_vec_size) {
      int gpu_id = bid / batch_size_per_gpu;     // target gpu id
      int sample_id = bid % batch_size_per_gpu;  // sample id on target gpu

      for (int i = 0; i < slot_num_per_gpu; i++) {
        int feature_row_index = bid * slot_num_per_gpu + i;
        TypeKey value_offset = row_offset[feature_row_index];
        TypeKey feature_num =
            row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

        float sum = 0.0f;
        for (int j = 0; j < feature_num; j++) {
          size_t value_index = hash_value_index[value_offset + j];
          sum += hash_table_value[value_index * embedding_vec_size + tid];
        }

        int slot_id =
            i * gpu_num + local_gpu_id;  // slot id on target gpu (for localizedSlotEmbedding)
        size_t feature_id = sample_id * slot_num + slot_id;  // feature id on target gpu

        // store the embedding vector
        embedding_features[gpu_id][feature_id * embedding_vec_size + tid] =
            TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
      }
    }
  }
}

// overload for fp16
template <typename TypeKey>
__global__ void forward_sum_fuse_align2_kernel(
    size_t local_gpu_id, size_t gpu_num, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, const TypeKey *row_offset,
    const size_t *hash_value_index, const float *hash_table_value, __half **embedding_features) {
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  int offset = (local_gpu_id + 1) * batch_size_per_gpu;
  for (int bid_offset = (blockIdx.x + offset); bid_offset < (batch_size + offset);
       bid_offset += gridDim.x) {
    int bid = bid_offset % batch_size;

    if (tid < embedding_vec_size) {
      const float2 *hash_table_value2 = reinterpret_cast<const float2 *>(hash_table_value);
      __half2 **embedding_features2 = reinterpret_cast<__half2 **>(embedding_features);

      int gpu_id = bid / batch_size_per_gpu;     // target gpu id
      int sample_id = bid % batch_size_per_gpu;  // sample id on target gpu

      for (int i = 0; i < slot_num_per_gpu; i++) {
        int feature_row_index = bid * slot_num_per_gpu + i;
        TypeKey value_offset = row_offset[feature_row_index];
        TypeKey feature_num =
            row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

        // use float type to do accumulation
        float2 sum2 = {0.0f, 0.0f};
        for (int j = 0; j < feature_num; j++) {
          size_t value_index = hash_value_index[value_offset + j];
          sum2.x += hash_table_value2[value_index * embedding_vec_size + tid].x;
          sum2.y += hash_table_value2[value_index * embedding_vec_size + tid].y;
        }
        __half2 sum = __float22half2_rn(sum2);

        int slot_id =
            i * gpu_num + local_gpu_id;  // slot id on target gpu (for localizedSlotEmbedding)
        size_t feature_id = sample_id * slot_num + slot_id;  // feature id on target gpu

        // store the embedding vector
        embedding_features2[gpu_id][feature_id * embedding_vec_size + tid] = sum;
      }
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_fuse(size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
                  size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, TypeEmbeddingComp **embedding_features,
                  size_t sm_count, cudaStream_t stream) {
  const size_t block_size = embedding_vec_size;
  int maxActiveBlocks;
  HCTR_LIB_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, forward_sum_fuse_kernel<TypeHashKey, TypeEmbeddingComp>, block_size, 0));
  const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

  forward_sum_fuse_kernel<<<grid_size, block_size, 0, stream>>>(
      id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
      embedding_vec_size, row_offset, hash_value_index, hash_table_value, embedding_features);
}

template <typename TypeHashKey>
void forward_fuse(size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
                  size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, __half **embedding_features, size_t sm_count,
                  cudaStream_t stream) {
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    int maxActiveBlocks;
    HCTR_LIB_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, forward_sum_fuse_align2_kernel<TypeHashKey>, block_size, 0));
    const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

    forward_sum_fuse_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
        embedding_vec_size / 2, row_offset, hash_value_index, hash_table_value, embedding_features);

  } else {
    const size_t block_size = embedding_vec_size;
    int maxActiveBlocks;
    HCTR_LIB_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, forward_sum_fuse_kernel<TypeHashKey, __half>, block_size, 0));
    const size_t grid_size = min(batch_size, sm_count * static_cast<size_t>(maxActiveBlocks));

    forward_sum_fuse_kernel<<<grid_size, block_size, 0, stream>>>(
        id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
        embedding_vec_size, row_offset, hash_value_index, hash_table_value, embedding_features);
  }
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_fuse_per_gpu(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<TypeHashKey> &row_offset, const Tensor2<size_t> &hash_value_index,
    const Tensor2<float> &hash_table_value, Tensor2<TypeEmbeddingComp *> &embedding_features,
    size_t sm_count, cudaStream_t stream) {
  if (combiner == 0) {
    forward_fuse(id, local_gpu_count, batch_size, batch_size_per_gpu, slot_num, slot_num_per_gpu,
                 embedding_vec_size, row_offset.get_ptr(), hash_value_index.get_ptr(),
                 hash_table_value.get_ptr(), embedding_features.get_ptr(), sm_count, stream);

  } else {
    HCTR_OWN_THROW(Error_t::WrongInput, "Invalid combiner type ");
  }

  return;
}

template void SparseEmbeddingFunctors::forward_fuse_per_gpu<unsigned int, float>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<unsigned int> &row_offset, const Tensor2<size_t> &hash_value_index,
    const Tensor2<float> &hash_table_value, Tensor2<float *> &embedding_features, size_t sm_count,
    cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_fuse_per_gpu<long long, float>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<long long> &row_offset, const Tensor2<size_t> &hash_value_index,
    const Tensor2<float> &hash_table_value, Tensor2<float *> &embedding_features, size_t sm_count,
    cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_fuse_per_gpu<unsigned int, __half>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<unsigned int> &row_offset, const Tensor2<size_t> &hash_value_index,
    const Tensor2<float> &hash_table_value, Tensor2<__half *> &embedding_features, size_t sm_count,
    cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_fuse_per_gpu<long long, __half>(
    size_t id, size_t local_gpu_count, size_t batch_size, size_t batch_size_per_gpu,
    size_t slot_num, size_t slot_num_per_gpu, size_t embedding_vec_size, int combiner,
    const Tensor2<long long> &row_offset, const Tensor2<size_t> &hash_value_index,
    const Tensor2<float> &hash_table_value, Tensor2<__half *> &embedding_features, size_t sm_count,
    cudaStream_t stream);

}  // namespace HugeCTR