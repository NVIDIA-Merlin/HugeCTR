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
#include "HugeCTR/include/pinned_buffer.hpp"

namespace HugeCTR {

template <typename TypeHashKey>
void SparseEmbeddingFunctors::init_embedding_per_slot(size_t slot_id, size_t slot_size,
                                                      size_t embedding_vec_size, size_t key_offset,
                                                      size_t value_index_offset,
                                                      float *embedding_table,
                                                      HashTable<TypeHashKey, size_t> &hash_table,
                                                      size_t *slot_ids, cudaStream_t stream) {
  TypeHashKey *hash_keys;
  CK_CUDA_THROW_(cudaMalloc(&hash_keys, slot_size * sizeof(TypeHashKey)));
  size_t *hash_value_indices;
  CK_CUDA_THROW_(cudaMalloc(&hash_value_indices, slot_size * sizeof(size_t)));
  PinnedBuffer<float> embedding_init(slot_size * embedding_vec_size);

  float up_bound = sqrt(1.f / slot_size);
  HugeCTR::UnifiedDataSimulator<float> fdata_sim(-up_bound, up_bound);

  for (size_t i = 0; i < (slot_size * embedding_vec_size); i++) {
    embedding_init.get()[i] = fdata_sim.get_num();
  }
  CK_CUDA_THROW_(cudaMemcpyAsync(embedding_table, embedding_init.get(),
                                 slot_size * embedding_vec_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

  memset_liner(hash_keys, (TypeHashKey)key_offset, (TypeHashKey)1, slot_size, stream);
  memset_liner(hash_value_indices, value_index_offset, 1ul, slot_size, stream);
  hash_table.insert(hash_keys, hash_value_indices, slot_size, stream);
  size_t value_head = hash_table.get_and_add_value_head(slot_size, stream);

  memset_const(slot_ids, slot_id, slot_size, stream);

  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  CK_CUDA_THROW_(cudaFree(hash_keys));
  CK_CUDA_THROW_(cudaFree(hash_value_indices));
}

template <typename TypeHashKey>
void SparseEmbeddingFunctors::init_embedding_per_slot(size_t slot_id, size_t slot_size,
                                                      size_t embedding_vec_size, size_t key_offset,
                                                      size_t value_index_offset,
                                                      float *embedding_table, size_t *slot_ids,
                                                      cudaStream_t stream) {
  TypeHashKey *hash_keys;
  CK_CUDA_THROW_(cudaMalloc(&hash_keys, slot_size * sizeof(TypeHashKey)));
  size_t *hash_value_indices;
  CK_CUDA_THROW_(cudaMalloc(&hash_value_indices, slot_size * sizeof(size_t)));
  PinnedBuffer<float> embedding_init(slot_size * embedding_vec_size);

  float up_bound = sqrt(1.f / slot_size);
  HugeCTR::UnifiedDataSimulator<float> fdata_sim(-up_bound, up_bound);

  for (size_t i = 0; i < (slot_size * embedding_vec_size); i++) {
    embedding_init.get()[i] = fdata_sim.get_num();
  }
  CK_CUDA_THROW_(cudaMemcpyAsync(embedding_table, embedding_init.get(),
                                 slot_size * embedding_vec_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

  memset_liner(hash_keys, (TypeHashKey)key_offset, (TypeHashKey)1, slot_size, stream);
  memset_liner(hash_value_indices, value_index_offset, 1ul, slot_size, stream);

  memset_const(slot_ids, slot_id, slot_size, stream);

  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  CK_CUDA_THROW_(cudaFree(hash_keys));
  CK_CUDA_THROW_(cudaFree(hash_value_indices));
}

template <typename TypeHashKey>
void SparseEmbeddingFunctors::init_embedding_per_gpu(
    size_t lid, size_t gid, size_t total_gpu_count, const std::vector<size_t> &slot_sizes,
    size_t embedding_vec_size, float *embedding_table, HashTable<TypeHashKey, size_t> &hash_table,
    size_t *slot_ids, const GPUResourceGroup &device_resources) {
  CudaDeviceContext context(device_resources[lid].get_device_id());

  size_t key_offset = 0;
  size_t value_index_offset = 0;
  for (size_t i = 0; i < slot_sizes.size(); i++) {
    size_t slot_size = slot_sizes[i];
    if ((i % total_gpu_count) == gid) {
      MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
               std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
               ", key_offset=" + std::to_string(key_offset) +
               ", value_index_offset=" + std::to_string(value_index_offset));
      init_embedding_per_slot(i, slot_size, embedding_vec_size, key_offset, value_index_offset,
                              embedding_table, hash_table, slot_ids,
                              device_resources[lid].get_stream());
      value_index_offset += slot_size;
      embedding_table += slot_size * embedding_vec_size;
      slot_ids += slot_size;
    }
    key_offset += slot_size;
  }
}

template <typename TypeHashKey>
void SparseEmbeddingFunctors::init_embedding_per_gpu(size_t lid, size_t gid, size_t total_gpu_count,
                                                     const std::vector<size_t> &slot_sizes,
                                                     size_t embedding_vec_size,
                                                     float *embedding_table, size_t *slot_ids,
                                                     const GPUResourceGroup &device_resources) {
  CudaDeviceContext context(device_resources[lid].get_device_id());

  size_t key_offset = 0;
  size_t value_index_offset = 0;
  for (size_t i = 0; i < slot_sizes.size(); i++) {
    size_t slot_size = slot_sizes[i];
    if ((i % total_gpu_count) == gid) {
      MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
               std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
               ", key_offset=" + std::to_string(key_offset) +
               ", value_index_offset=" + std::to_string(value_index_offset));
      init_embedding_per_slot<TypeHashKey>(i, slot_size, embedding_vec_size, key_offset,
                                           value_index_offset, embedding_table, slot_ids,
                                           device_resources[lid].get_stream());
      value_index_offset += slot_size;
      embedding_table += slot_size * embedding_vec_size;
      slot_ids += slot_size;
    }
    key_offset += slot_size;
  }
}

template void SparseEmbeddingFunctors::init_embedding_per_gpu<unsigned int>(
    size_t lid, size_t gid, size_t total_gpu_count, const std::vector<size_t> &slot_sizes,
    size_t embedding_vec_size, float *embedding_table, HashTable<unsigned int, size_t> &hash_table,
    size_t *slot_ids, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::init_embedding_per_gpu<long long>(
    size_t lid, size_t gid, size_t total_gpu_count, const std::vector<size_t> &slot_sizes,
    size_t embedding_vec_size, float *embedding_table, HashTable<long long, size_t> &hash_table,
    size_t *slot_ids, const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::init_embedding_per_gpu<unsigned int>(
    size_t lid, size_t gid, size_t total_gpu_count, const std::vector<size_t> &slot_sizes,
    size_t embedding_vec_size, float *embedding_table, size_t *slot_ids,
    const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::init_embedding_per_gpu<long long>(
    size_t lid, size_t gid, size_t total_gpu_count, const std::vector<size_t> &slot_sizes,
    size_t embedding_vec_size, float *embedding_table, size_t *slot_ids,
    const GPUResourceGroup &device_resources);

}  // namespace HugeCTR