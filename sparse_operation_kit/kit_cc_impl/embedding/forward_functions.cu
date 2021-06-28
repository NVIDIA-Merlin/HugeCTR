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

#include "embeddings/forward_functions.h"
#include "embeddings/forward_functions.cuh"

namespace SparseOperationKit {

void get_hash_value(size_t count, size_t embedding_vec_size, const size_t *value_index,
                    const float *embedding_table, float *value_retrieved,
                    cudaStream_t stream) {
    const size_t block_size = embedding_vec_size;
    const size_t grid_size = count;

    HugeCTR::get_hash_value_kernel<<<grid_size, block_size, 0, stream>>>(count, embedding_vec_size,
                                                    value_index, embedding_table, value_retrieved);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream) {
    HugeCTR::forward_sum(batch_size, slot_num, embedding_vec_size,
                         row_offset, hash_value_index, hash_table_value,
                         embedding_feature, stream);
}

template void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const int64_t *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, float *embedding_feature,
                 cudaStream_t stream);

// template <typename TypeHashKey, typename TypeEmbeddingComp>
// void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const TypeHashKey *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
//                   cudaStream_t stream) {
//     HugeCTR::forward_mean(batch_size, slot_num, embedding_vec_size,
//                           row_offset, hash_value_index, hash_table_value,
//                           embedding_feature, stream);
// }

// template void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
//                   const long long *row_offset, const size_t *hash_value_index,
//                   const float *hash_table_value, float *embedding_feature,
//                   cudaStream_t stream);

template <typename TypeKey, typename TypeEmbeddingComp>
void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const TypeKey *row_offset, TypeEmbeddingComp *embedding_feature,
                      cudaStream_t stream) {
    HugeCTR::do_forward_scale(batchsize_per_gpu, slot_num, embedding_vec_size,
                              row_offset, embedding_feature, stream);
}

template void do_forward_scale(size_t batchsize_per_gpu, size_t slot_num, size_t embedding_vec_size,
                      const int64_t *row_offset, float *embedding_feature,
                      cudaStream_t stream);

template <typename Type>
void memset_liner(Type *data, Type start_value, Type stride_value,
                  size_t n, cudaStream_t stream) {
    HugeCTR::memset_liner(data, start_value, stride_value, n, stream);
}

template void memset_liner(size_t *data, size_t start_value, size_t stride_value,
                  size_t n, cudaStream_t stream);

} // namespace SparseOperationKit