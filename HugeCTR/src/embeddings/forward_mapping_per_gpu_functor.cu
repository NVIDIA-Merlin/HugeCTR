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

namespace HugeCTR {

namespace {

// for one-hot, the value_index mapping is linear (no need to use hashtable)
template <typename TypeKey>
__global__ void hash_key_value_index_mapping_kernel(size_t nnz, int slot_num,
                                                    const uint32_t *mapping_offsets,
                                                    const TypeKey *hash_key,
                                                    size_t *hash_value_index) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nnz) {
    int slot_id = gid % slot_num;
    hash_value_index[gid] = hash_key[gid] - mapping_offsets[slot_id];
  }
}

}  // namespace

/**
 * forward propagation on each GPU for LocalizedSlotSparseEmbeddingOneHot.
 * Because there is no hashtable in this class, so there must be a mapping table
 * between input valud_index and local value_index.
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots for current GPU
 * @param row_offset row_offset (CSR format of input sparse tensors)
 * @param hash_key value (CSR format of input sparse tensors)
 * @param nnz non-zero feature number per batch
 * @param mapping_offsets the mapping between input value_index and local value_index
 * @param hash_value_index hash table value_index(row index of embedding)
 * @param stream cuda stream
 */
template <typename TypeHashKey>
void SparseEmbeddingFunctors::forward_mapping_per_gpu(size_t batch_size, size_t slot_num,
                                                      const Tensor2<TypeHashKey> &hash_key,
                                                      size_t nnz,
                                                      const Tensor2<uint32_t> &mapping_offsets,
                                                      Tensor2<size_t> &hash_value_index,
                                                      cudaStream_t stream) {
  // remove hashtable get_insert(), and do linear mapping between key and value_index
  if (nnz > 0) {
    hash_key_value_index_mapping_kernel<<<(nnz + 255) / 256, 256, 0, stream>>>(
        nnz, slot_num, mapping_offsets.get_ptr(), hash_key.get_ptr(), hash_value_index.get_ptr());
  }

  return;
}

template void SparseEmbeddingFunctors::forward_mapping_per_gpu<unsigned int>(
    size_t batch_size, size_t slot_num, const Tensor2<unsigned int> &hash_key, size_t nnz,
    const Tensor2<uint32_t> &mapping_offsets, Tensor2<size_t> &hash_value_index,
    cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_mapping_per_gpu<long long>(
    size_t batch_size, size_t slot_num, const Tensor2<long long> &hash_key, size_t nnz,
    const Tensor2<uint32_t> &mapping_offsets, Tensor2<size_t> &hash_value_index,
    cudaStream_t stream);

}  // namespace HugeCTR
