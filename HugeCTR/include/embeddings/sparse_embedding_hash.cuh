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

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "HugeCTR/include/common.hpp"
#include "cub/cub/device/device_radix_sort.cuh"

#include "HugeCTR/include/hashtable/nv_hashtable.cuh"

// using namespace cooperative_groups;

namespace HugeCTR {

namespace SparseEmbeddingHashKernels {

//---------------------------------GPU kernel functions--------------------------------------
// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeHashKey, typename TypeHashValueIndex>
__global__ void forward_sum_kernel(const int batch_size, const int slot_num,
                                   const int embedding_vec_size, const TypeHashKey *row_offset,
                                   const TypeHashValueIndex *hash_value_index,
                                   const float *hash_table_value, float *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeHashKey value_offset = row_offset[feature_row_index];
      TypeHashKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; j++) {
        TypeHashValueIndex value_index = hash_value_index[value_offset + j];
        sum += hash_table_value[value_index * embedding_vec_size + tid];

        // just for debug
        // printf("bid=%d, slot=%d, tid=%d, j=%d, value_index=%d, value=%f\n", bid, i, tid, j,
        // value_index, hash_table_value[value_index * embedding_vec_size + tid]);
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] = sum;
    }
  }
}

// forward kernel function: this is an additional function for combiner=mean
template <typename TypeHashKey>
__global__ void forward_scale_kernel(const int batch_size, const int slot_num,
                                     const int embedding_vec_size, const TypeHashKey *row_offset,
                                     float *embedding_feature) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      int feature_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      int feature_index = feature_row_index * embedding_vec_size + tid;
      float feature = embedding_feature[feature_index];
      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / (float)feature_num;
      }

      embedding_feature[feature_index] = feature * scaler;
    }
  }
}

// backward kernel function: for combiner=sum
template <typename TypeHashKey>
__global__ void backward_sum_kernel(const int batch_size, const int slot_num,
                                    const int embedding_vec_size, const float *top_grad,
                                    float *wgrad) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_index = (bid * slot_num + i) * embedding_vec_size + tid;
      wgrad[feature_index] = top_grad[feature_index];
    }
  }
}

// backward kernel function: for combiner=mean
template <typename TypeHashKey>
__global__ void backward_mean_kernel(const int batch_size, const int slot_num,
                                     const int embedding_vec_size, const TypeHashKey *row_offset,
                                     const float *top_grad, float *wgrad) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeHashKey value_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
      int feature_index = feature_row_index * embedding_vec_size + tid;
      float scaler = 1.0f;
      if (value_num > 1) {
        scaler = 1.0f / (float)value_num;  // partial derivatice of MEAN
      }

      float grad = top_grad[feature_index];
      wgrad[feature_index] = scaler * grad;
    }
  }
}

// expand sample id by row_offset
template <typename TypeHashKey>
__global__ void sample_id_expand_kernel(const int batch_size, const int slot_num,
                                        const TypeHashKey *row_offset, TypeHashKey *sample_id) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < (batch_size * slot_num)) {
    TypeHashKey offset = row_offset[gid];
    int value_num = row_offset[gid + 1] - offset;
    for (int i = 0; i < value_num; i++) {
      sample_id[offset + i] = gid;
    }
  }
}

template <typename TypeHashKey>
__device__ __forceinline__ void swap(TypeHashKey &a, TypeHashKey &b) {
  TypeHashKey temp = a;
  a = b;
  b = temp;
}

// count the number for each unduplicated hash_value_index
template <typename TypeHashValueIndex>
__global__ void value_count_kernel(const int nnz, const TypeHashValueIndex *hash_value_index_sort,
                                   uint32_t *hash_value_index_count,
                                   uint32_t *hash_value_index_count_offset,
                                   uint32_t *hash_value_index_count_counter) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < nnz) {
    TypeHashValueIndex cur_value = hash_value_index_sort[gid];
    uint32_t sample_num = 0;
    if (gid > 0) {
      TypeHashValueIndex former_value = hash_value_index_sort[gid - 1];
      // decide if this is the start of a group(the elements in this group have the same
      // hash_value_index_sort)
      if (cur_value != former_value) {
        sample_num = 1;
      } else {
        sample_num = 0;
      }
    } else {  // gid == 0
      sample_num = 1;
    }

    // if sample_num > 0, continue to compare the current hash_value_index_sort with latter values,
    // cal how many elements in this group
    if (sample_num) {
      while (gid + sample_num < nnz) {
        TypeHashValueIndex latter_value = hash_value_index_sort[gid + sample_num];
        if (cur_value == latter_value) {
          sample_num++;
        } else {
          break;
        }
      }

      // record sample_num in the hash_value_index_count array(with all non-zero values) and the
      // corresponding offset in the hash_value_index_count_offset array This is a parallel writing,
      // so the hash_value_index_count array will be out-of-order(not sorted like the original
      // hash_value_index_sort array).
      uint32_t counter = atomicAdd(hash_value_index_count_counter, 1);
      hash_value_index_count[counter] = sample_num;
      hash_value_index_count_offset[counter] = gid;
    }
  }
}

template <typename TypeHashKey, typename TypeHashValueIndex>
__global__ void opt_adam_kernel(const uint32_t hash_value_index_count_num,
                                const int embedding_vec_size, const AdamOptHyperParams adam,
                                const TypeHashKey *sample_id,
                                const TypeHashValueIndex *hash_value_index_sort,
                                const uint32_t *hash_value_index_count,
                                const uint32_t *hash_value_index_count_offset, const float *wgrad,
                                TypeHashValueIndex *deltaw_hash_value_index, float *deltaw) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t sample_num = hash_value_index_count[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }

    // compute the grad of the weights and update it
    TypeHashValueIndex row_index = hash_value_index_sort[offset];
    TypeHashValueIndex feature_index = row_index * embedding_vec_size + tid;
    float mi = adam.beta1 * adam.m_ptr[feature_index] + (1.0f - adam.beta1) * gi;
    float vi = adam.beta2 * adam.v_ptr[feature_index] + (1.0f - adam.beta2) * gi * gi;
    adam.m_ptr[feature_index] = mi;
    adam.v_ptr[feature_index] = vi;
    float weight_diff = -adam.alpha_t * mi / (sqrtf(vi) + adam.epsilon);

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = weight_diff;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

template <typename TypeHashKey, typename TypeHashValueIndex>
__global__ void opt_momentum_sgd_kernel(
    const uint32_t hash_value_index_count_num, const int embedding_vec_size, const float lr,
    const MomentumSgdOptHyperParams momentum, const TypeHashKey *sample_id,
    const TypeHashValueIndex *hash_value_index_sort, const uint32_t *hash_value_index_count,
    const uint32_t *hash_value_index_count_offset, const float *wgrad,
    TypeHashValueIndex *deltaw_hash_value_index, float *deltaw) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t sample_num = hash_value_index_count[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }

    // compute the grad of the weights and update it
    TypeHashValueIndex row_index = hash_value_index_sort[offset];
    TypeHashValueIndex feature_index = row_index * embedding_vec_size + tid;
    float mo = momentum.factor * momentum.momentum_ptr[feature_index] - lr * gi;
    momentum.momentum_ptr[feature_index] = mo;

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = mo;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

template <typename TypeHashKey, typename TypeHashValueIndex>
__global__ void opt_nesterov_kernel(
    const uint32_t hash_value_index_count_num, const int embedding_vec_size, const float lr,
    const NesterovOptHyperParams nesterov, const TypeHashKey *sample_id,
    const TypeHashValueIndex *hash_value_index_sort, const uint32_t *hash_value_index_count,
    const uint32_t *hash_value_index_count_offset, const float *wgrad,
    TypeHashValueIndex *deltaw_hash_value_index, float *deltaw) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t sample_num = hash_value_index_count[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }

    // compute the grad of the weights and update it
    TypeHashValueIndex row_index = hash_value_index_sort[offset];
    TypeHashValueIndex feature_index = row_index * embedding_vec_size + tid;
    float accm_old = nesterov.accm_ptr[feature_index];
    float accm_new = nesterov.mu * accm_old - lr * gi;
    nesterov.accm_ptr[feature_index] = accm_new;
    float weight_diff = -nesterov.mu * accm_old + (1.0f + nesterov.mu) * accm_new;

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = weight_diff;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

template <typename TypeHashValueIndex>
__global__ void update_kernel(const uint32_t hash_value_index_count_num,
                              const int embedding_vec_size,
                              TypeHashValueIndex *deltaw_hash_value_index, float *deltaw,
                              float *hash_table_value) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if ((bid < hash_value_index_count_num) && (tid < embedding_vec_size)) {
    TypeHashValueIndex value_index = deltaw_hash_value_index[bid];
    long long feature_index = value_index * embedding_vec_size + tid;

    hash_table_value[feature_index] += deltaw[bid * embedding_vec_size + tid];
  }
}

template <typename Type>
__global__ void memset_liner(Type *data, Type start_value, Type stride_value, long long n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = start_value + (Type)gid * stride_value;
  }
}

template <typename TypeHashValueIndex>
__global__ void get_hash_table_value(const long long count, const int embedding_vec_size,
                                     const TypeHashValueIndex *value_index,
                                     const float *hash_table_value, float *value_retrieved) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < count && tid < embedding_vec_size) {
    TypeHashValueIndex index = value_index[bid];
    value_retrieved[bid * embedding_vec_size + tid] =
        hash_table_value[index * embedding_vec_size + tid];
  }
}

//-----------------------------------host functions wrappers for GPU
//-------------------------------------------
template <typename TypeHashKey, typename TypeHashValueIndex>
void do_forward(const cudaStream_t stream, const int batch_size, const int slot_num,
                const int embedding_vec_size, const TypeHashKey *row_offset,
                const TypeHashKey *hash_key,
                const nv::HashTable<TypeHashKey, TypeHashValueIndex,
                                    std::numeric_limits<TypeHashKey>::max()> *hash_table,
                const float *hash_table_value, TypeHashValueIndex *hash_value_index,
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
void do_forward_scale(const cudaStream_t stream, const int batch_size, const int slot_num,
                      const int embedding_vec_size, const TypeHashKey *row_offset,
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

template <typename TypeHashKey>
void do_backward(const cudaStream_t stream, const int batch_size, const int slot_num,
                 const int embedding_vec_size, const int combiner, const TypeHashKey *row_offset,
                 const float *top_grad, float *wgrad) {
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

template <typename TypeHashKey, typename TypeHashValueIndex>
void do_update_params(
    const cudaStream_t stream, const int batch_size, const int slot_num,
    const int embedding_vec_size, const long long max_vocabulary_size_per_gpu, OptParams opt_params,
    const TypeHashKey *row_offset, const TypeHashKey *hash_key,
    const nv::HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>
        *hash_table,
    TypeHashValueIndex *hash_value_index, TypeHashKey *sample_id, TypeHashKey *sample_id_sort,
    TypeHashValueIndex *hash_value_index_sort, uint32_t *hash_value_index_count,
    uint32_t *hash_value_index_count_offset, uint32_t *hash_value_index_count_counter,
    void *temp_storage_sort, size_t temp_storage_sort_bytes, const float *wgrad,
    TypeHashValueIndex *deltaw_hash_value_index, float *deltaw, float *hash_table_value) {
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

template <typename Type>
void do_memset_liner(cudaStream_t stream, Type *data, Type start_value, Type stride_value,
                     long long n) {
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
void do_get_hash_table_value(const cudaStream_t stream, const long long count,
                             const int embedding_vec_size, const TypeHashValueIndex *value_index,
                             const float *hash_table_value, float *value_retrieved) {
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

}  // end of namespace SparseEmbeddingHashKernels

}  // end of namespace HugeCTR
