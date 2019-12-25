/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#pragma once

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/hashtable/nv_hashtable.cuh"
#include "HugeCTR/include/embeddings/sparse_embedding_kernels.cuh"

namespace HugeCTR {

// forward computation 
template <typename TypeHashKey, typename TypeHashValueIndex>
void do_forward(const cudaStream_t stream, 
                const int batch_size, 
                const int slot_num,
                const int embedding_vec_size, 
                const TypeHashKey *row_offset,
                const TypeHashKey *hash_key,
                const nv::HashTable<TypeHashKey, TypeHashValueIndex,
                                    std::numeric_limits<TypeHashKey>::max()> *hash_table,
                const float *hash_table_value, 
                TypeHashValueIndex *hash_value_index,
                float *embedding_feature) {
  try {
    // get hash_value_index from hash_table by hash_key
    size_t num;
    CK_CUDA_THROW_(cudaMemcpyAsync(&num, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                  cudaMemcpyDeviceToHost, stream));
    hash_table->get_insert(hash_key, hash_value_index, num, stream);
    // hash_table->get(hash_key, hash_value_index, num, stream);

    // do sum reduction
    dim3 blockSize(embedding_vec_size, 1,
                  1);  // each thread corresponds to one element in a embedding vector
    dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
    forward_sum_kernel<TypeHashKey, TypeHashValueIndex>
        <<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num, embedding_vec_size, row_offset,
                                            hash_value_index, hash_table_value, embedding_feature);
    // for combiner=mean, call do_forward_scale() after this do_forward() and NCCL all-reduce
    // operation
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

// this is an additional function for combiner=mean
template <typename TypeHashKey>
void do_forward_scale(const cudaStream_t stream, 
                      const int batch_size, 
                      const int slot_num,
                      const int embedding_vec_size, 
                      const TypeHashKey *row_offset,
                      float *embedding_feature) {
  try {
    dim3 blockSize(embedding_vec_size, 1, 1);
    dim3 gridSize(batch_size, 1, 1);

    forward_scale_kernel<<<gridSize, blockSize, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, row_offset, embedding_feature);

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

// calculate wgrad
template <typename TypeHashKey>
void do_backward(const cudaStream_t stream, 
                const int batch_size, 
                const int slot_num,
                const int embedding_vec_size, 
                const int combiner, 
                const TypeHashKey *row_offset,
                const float *top_grad, 
                float *wgrad) {
  try {
    dim3 blockSize(embedding_vec_size, 1,
                  1);                // each thread corresponds to one element in a embedding vetor
    dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample

    if (combiner == 0)  // sum
    {
      backward_sum_kernel<TypeHashKey><<<gridSize, blockSize, 0, stream>>>(
          batch_size, slot_num, embedding_vec_size, top_grad, wgrad);
    } else if (combiner == 1)  // mean
    {
      backward_mean_kernel<<<gridSize, blockSize, 0, stream>>>(
          batch_size, slot_num, embedding_vec_size, row_offset, top_grad, wgrad);
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

// update embedding table(weights)
template <typename TypeHashKey, typename TypeHashValueIndex>
void do_update_params(const cudaStream_t stream, 
                      const int batch_size, 
                      const int slot_num,
                      const int embedding_vec_size, 
                      const long long max_vocabulary_size_per_gpu, 
                      OptParams opt_params,
                      const TypeHashKey *row_offset, 
                      const TypeHashKey *hash_key,
                      const nv::HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>
                          *hash_table,
                      TypeHashValueIndex *hash_value_index, 
                      TypeHashKey *sample_id, 
                      TypeHashKey *sample_id_sort,
                      TypeHashValueIndex *hash_value_index_sort, 
                      uint32_t *hash_value_index_count,
                      uint32_t *hash_value_index_count_offset, 
                      uint32_t *hash_value_index_count_counter,
                      void *temp_storage_sort, 
                      size_t temp_storage_sort_bytes, 
                      const float *wgrad,
                      TypeHashValueIndex *deltaw_hash_value_index, 
                      float *deltaw, 
                      float *hash_table_value) {
  try {
    // step1: expand sample IDs
    dim3 blockSize(64, 1, 1);
    dim3 gridSize((batch_size * slot_num + blockSize.x - 1) / blockSize.x, 1, 1);
    sample_id_expand_kernel<<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num, row_offset,
                                                                sample_id);

    int nnz;
    // this async memcpy will not perform as a async operation because the host memory is not a
    // pinned memory
    CK_CUDA_THROW_(cudaMemcpyAsync(&nnz, row_offset + batch_size * slot_num, sizeof(TypeHashKey),
                                  cudaMemcpyDeviceToHost, stream));

    // step2: get hash_value_index by hash_key
    hash_table->get_insert(hash_key, hash_value_index, nnz, stream);

    // step3: sort by hash_value_index
    int end_bit = (int)log2((float)max_vocabulary_size_per_gpu) + 1;
    CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
        (void *)temp_storage_sort, temp_storage_sort_bytes, hash_value_index, hash_value_index_sort,
        sample_id, sample_id_sort, nnz, 0, end_bit, stream, false));

    // step4: count the number for each unduplicated hash_value_index
    CK_CUDA_THROW_(cudaMemsetAsync(hash_value_index_count_counter, 0, sizeof(uint32_t), stream));
    gridSize.x = (nnz + (blockSize.x - 1)) / blockSize.x;
    value_count_kernel<<<gridSize, blockSize, 0, stream>>>(
        nnz, hash_value_index_sort, hash_value_index_count, hash_value_index_count_offset,
        hash_value_index_count_counter);

    uint32_t hash_hash_value_index_count_num = 0;
    // this async memcpy will not perform as a async operation because the host memory is not a
    // pinned memroy
    CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num, hash_value_index_count_counter,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    // step5: use optimizer method to compute deltaw, and record corresponding
    // deltaw_hash_value_index
    blockSize.x = embedding_vec_size;
    gridSize.x = max(1, hash_hash_value_index_count_num);
    switch (opt_params.optimizer) {
      case 0:  // adam
        opt_params.hyperparams.adam.alpha_t =
            opt_params.lr *
            sqrt(1 - pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
            (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

        opt_adam_kernel<<<gridSize, blockSize, 0, stream>>>(
            hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
            sample_id_sort, hash_value_index_sort, hash_value_index_count,
            hash_value_index_count_offset, wgrad, deltaw_hash_value_index, (float *)deltaw);
        break;
      case 1:  // momentum sgd
        opt_momentum_sgd_kernel<<<gridSize, blockSize, 0, stream>>>(
            hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
            opt_params.hyperparams.momentum, sample_id_sort, hash_value_index_sort,
            hash_value_index_count, hash_value_index_count_offset, wgrad, deltaw_hash_value_index,
            (float *)deltaw);
        break;
      case 2:  // nesterov
        opt_nesterov_kernel<<<gridSize, blockSize, 0, stream>>>(
            hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
            opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
            hash_value_index_count, hash_value_index_count_offset, wgrad, deltaw_hash_value_index,
            (float *)deltaw);
        break;
      default:
        CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
    }

    // step6: update hash_table_value by deltaw
    blockSize.x = embedding_vec_size;
    gridSize.x = max(1, hash_hash_value_index_count_num);
    update_kernel<TypeHashValueIndex>
        <<<gridSize, blockSize, 0, stream>>>(hash_hash_value_index_count_num, embedding_vec_size,
                                            deltaw_hash_value_index, deltaw, hash_table_value);
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

// set liner data for a buffer
template <typename Type>
void do_memset_liner(const cudaStream_t stream, 
                      Type *data, 
                      const Type start_value, 
                      const Type stride_value,
                      const long long n) {
  try {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    memset_liner<Type><<<gridSize, blockSize, 0, stream>>>(data, start_value, stride_value, n);
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

// get hash table value by value_index
template <typename TypeHashValueIndex>
void do_get_hash_table_value(const cudaStream_t stream, 
                              const long long count,
                              const int embedding_vec_size, 
                              const TypeHashValueIndex *value_index,
                              const float *hash_table_value, 
                              float *value_retrieved) {
  try {
    int blockSize = embedding_vec_size;
    int gridSize = count;

    get_hash_table_value<<<gridSize, blockSize, 0, stream>>>(count, embedding_vec_size, value_index,
                                                            hash_table_value, value_retrieved);

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

}  // end of namespace HugeCTR
