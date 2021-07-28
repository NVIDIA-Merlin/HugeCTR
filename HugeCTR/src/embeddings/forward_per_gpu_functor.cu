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
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {
/**
 * All the CUDA kernel functions used by embedding layer are defined in this file, including
 * forward propagation, backward propagation. The functions are defined by propagation type
 * and combiner type(sum or mean) as below:
 *   1) forward
 *        sum: calling forward_sum_kernel()
 *        mean: calling foward_sum_kernel() + forward_scale_kernel()
 *   2) backward:
 *        calculating wgrad:
 *          sum: calling backward_sum_kernel()
 *          mean: calling backward_mean_kernel()
 *        update embedding table: including several steps as below,
 *          step1: expand sample IDs, calling sample_id_expand_kernel()
 *          step2: get value_index by key (will call hash_table->get_insert() in nv_hashtable lib)
 *          step3: sort by value_index (will call cub::DeviceRadixSort::SortPairs in cub lib)
 *          step4: count the number for each unduplicated value_index, calling value_count_kernel()
 *          step5: use optimizer method to compute deltaw, and record corresponding, including three
 * types of optimizer: Adam: caling opt_adam_kernel() Momentum sgd: calling
 * opt_momentum_sgd_kernel() Nesterov: calling opt_nesterov_kernel() step6: update embedding table
 * by deltaw, calling update_kernel()
 */

// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_sum_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                   const TypeKey *row_offset, const size_t *hash_value_index,
                                   const float *hash_table_value,
                                   TypeEmbeddingComp *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum += (value_index != std::numeric_limits<size_t>::max())
                   ? hash_table_value[value_index * embedding_vec_size + tid]
                   : 0.0f;
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
    }
  }
}

template <typename TypeKey>
__global__ void forward_sum_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                          const TypeKey *row_offset, const size_t *hash_value_index,
                                          const float *hash_table_value,
                                          __half *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    const float2 *hash_table_value2 = reinterpret_cast<const float2 *>(hash_table_value);
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      // use float type to do accumulation
      float2 sum2 = {0.0f, 0.0f};
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum2.x += (value_index != std::numeric_limits<size_t>::max())
                      ? hash_table_value2[value_index * embedding_vec_size + tid].x
                      : 0.0f;
        sum2.y += (value_index != std::numeric_limits<size_t>::max())
                      ? hash_table_value2[value_index * embedding_vec_size + tid].y
                      : 0.0f;
      }
      __half2 sum = __float22half2_rn(sum2);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = sum;
    }
  }
}

// forward kernel funcion: for combiner=mean in LocalizedEmbedding
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_mean_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                    const TypeKey *row_offset, const size_t *hash_value_index,
                                    const float *hash_table_value,
                                    TypeEmbeddingComp *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      int feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum += (value_index != std::numeric_limits<size_t>::max())
                   ? hash_table_value[value_index * embedding_vec_size + tid]
                   : 0.0f;
      }

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / feature_num;
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum * scaler);
    }
  }
}

template <typename TypeKey>
__global__ void forward_mean_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                           const TypeKey *row_offset,
                                           const size_t *hash_value_index,
                                           const float *hash_table_value,
                                           __half *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    const float2 *hash_table_value2 = reinterpret_cast<const float2 *>(hash_table_value);
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      int feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      // use float to do accumulation
      float2 sum = {0.0f, 0.0f};
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum.x += (value_index != std::numeric_limits<size_t>::max())
                     ? hash_table_value2[value_index * embedding_vec_size + tid].x
                     : 0.0f;
        sum.y += (value_index != std::numeric_limits<size_t>::max())
                     ? hash_table_value2[value_index * embedding_vec_size + tid].y
                     : 0.0f;
      }
      __half2 sum2 = __float22half2_rn(sum);

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / feature_num;
      }
      __half2 scaler2 = __float2half2_rn(scaler);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = __hmul2(sum2, scaler2);
    }
  }
}

// do sum reduction
template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size =
      embedding_vec_size;  // each thread corresponds to one element in an embedding vector
  forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num, embedding_vec_size,
                                                           row_offset, hash_value_index,
                                                           hash_table_value, embedding_feature);
}

// do sum reduction
template <typename TypeHashKey>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, __half *embedding_feature, cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
        hash_table_value, embedding_feature);
  } else {
    const size_t block_size =
        embedding_vec_size;  // each thread corresponds to one element in an embedding vector
    forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
        embedding_feature);
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                  cudaStream_t stream) {
  const size_t grid_size = batch_size;
  const size_t block_size = embedding_vec_size;
  forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
      embedding_feature);
}

template <typename TypeHashKey>
void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, __half *embedding_feature, cudaStream_t stream) {
  const size_t grid_size = batch_size;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
        hash_table_value, embedding_feature);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
        embedding_feature);
  }
}

}  // namespace

/**
 * forward propagation on each GPU for LocalizedSlotSparseEmbeddingHash
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots for current GPU
 * @param embedding_vec_size embedding vector size.
 * @param combiner 0-sum; 1-mean
 * @param row_offset row_offset (CSR format of input sparse tensors)
 * @param hash_key value (CSR format of input sparse tensors)
 * @param nnz non-zero feature number per batch
 * @param hash_table hash table, pairs of <key, value_index>
 * @param hash_table_value hash table value, which represents embedding vector
 * @param hash_value_index hash table value_index(row index of embedding)
 * @param embedding_feature embedding feature (output)
 * @param stream cuda stream
 */
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_per_gpu(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<TypeHashKey> &row_offset, const Tensor2<TypeHashKey> &hash_key, size_t nnz,
    HashTable<TypeHashKey, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<TypeEmbeddingComp> &embedding_feature,
    cudaStream_t stream) {
  try {
    if (train) {
      hash_table.get_insert(hash_key.get_ptr(), hash_value_index.get_ptr(), nnz, stream);
    } else {
      hash_table.get_mark(hash_key.get_ptr(), hash_value_index.get_ptr(), nnz, stream);
    }

    // do sum reduction
    if (combiner == 0) {
      forward_sum(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                  hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                  embedding_feature.get_ptr(), stream);
    } else if (combiner == 1) {
      forward_mean(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                   hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                   embedding_feature.get_ptr(), stream);
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::forward_per_gpu<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    HashTable<unsigned int, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    HashTable<long long, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    HashTable<unsigned int, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<__half> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    HashTable<long long, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<__half> &embedding_feature, cudaStream_t stream);

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_sum_per_gpu(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<TypeHashKey> &row_offset, const Tensor2<TypeHashKey> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<TypeEmbeddingComp> &embedding_feature, cudaStream_t stream) {
  try {
    // do sum reduction
    if (combiner == 0) {
      forward_sum(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                  hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                  embedding_feature.get_ptr(), stream);
    } else if (combiner == 1) {
      forward_mean(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                   hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                   embedding_feature.get_ptr(), stream);
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::forward_sum_per_gpu<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<__half> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<__half> &embedding_feature, cudaStream_t stream);

}  // namespace HugeCTR
