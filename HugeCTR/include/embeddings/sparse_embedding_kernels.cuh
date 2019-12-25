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

//#include <cooperative_groups.h>
//#include <cuda_runtime.h>
#include "HugeCTR/include/common.hpp"
#include "cub/cub/device/device_radix_sort.cuh"

//#include "HugeCTR/include/hashtable/nv_hashtable.cuh"

// using namespace cooperative_groups;

namespace HugeCTR {

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
 *          step5: use optimizer method to compute deltaw, and record corresponding, including three types of optimizer:
 *            Adam: caling opt_adam_kernel()
 *            Momentum sgd: calling opt_momentum_sgd_kernel()
 *            Nesterov: calling opt_nesterov_kernel()
 *          step6: update embedding table by deltaw, calling update_kernel()
 */

// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeKey, typename TypeValueIndex>
__global__ void forward_sum_kernel(const int batch_size, 
                                   const int slot_num,
                                   const int embedding_vec_size, 
                                   const TypeKey *row_offset,
                                   const TypeValueIndex *hash_value_index,
                                   const float *hash_table_value, 
                                   float *embedding_feature) {
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
        TypeValueIndex value_index = hash_value_index[value_offset + j];
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
template <typename TypeKey>
__global__ void forward_scale_kernel(const int batch_size, 
                                     const int slot_num,
                                     const int embedding_vec_size, 
                                     const TypeKey *row_offset,
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
template <typename TypeKey>
__global__ void backward_sum_kernel(const int batch_size, 
                                    const int slot_num,
                                    const int embedding_vec_size, 
                                    const float *top_grad,
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
template <typename TypeKey>
__global__ void backward_mean_kernel(const int batch_size, 
                                     const int slot_num,
                                     const int embedding_vec_size, 
                                     const TypeKey *row_offset,
                                     const float *top_grad, 
                                     float *wgrad) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_num = row_offset[feature_row_index + 1] - row_offset[feature_row_index];
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
template <typename TypeKey>
__global__ void sample_id_expand_kernel(const int batch_size, 
                                        const int slot_num,
                                        const TypeKey *row_offset, 
                                        TypeKey *sample_id) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < (batch_size * slot_num)) {
    TypeKey offset = row_offset[gid];
    int value_num = row_offset[gid + 1] - offset;
    for (int i = 0; i < value_num; i++) {
      sample_id[offset + i] = gid;
    }
  }
}

template <typename TypeKey>
__device__ __forceinline__ void swap(TypeKey &a, TypeKey &b) {
  TypeKey temp = a;
  a = b;
  b = temp;
}

// count the number for each unduplicated hash_value_index
template <typename TypeValueIndex>
__global__ void value_count_kernel(const int nnz, 
                                   const TypeValueIndex *hash_value_index_sort,
                                   uint32_t *hash_value_index_count,
                                   uint32_t *hash_value_index_count_offset,
                                   uint32_t *hash_value_index_count_counter) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < nnz) {
    TypeValueIndex cur_value = hash_value_index_sort[gid];
    uint32_t sample_num = 0;
    if (gid > 0) {
      TypeValueIndex former_value = hash_value_index_sort[gid - 1];
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
        TypeValueIndex latter_value = hash_value_index_sort[gid + sample_num];
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

// calculate weights update value(deltaw) by adam opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_adam_kernel(const uint32_t hash_value_index_count_num,
                                const int embedding_vec_size, 
                                const AdamOptHyperParams adam,
                                const TypeKey *sample_id,
                                const TypeValueIndex *hash_value_index_sort,
                                const uint32_t *hash_value_index_count,
                                const uint32_t *hash_value_index_count_offset, 
                                const float *wgrad,
                                TypeValueIndex *deltaw_hash_value_index, 
                                float *deltaw) {
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
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
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

// calculate weights update value(deltaw) by momentum_sgd opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_momentum_sgd_kernel(const uint32_t hash_value_index_count_num, 
                                        const int embedding_vec_size, 
                                        const float lr,
                                        const MomentumSgdOptHyperParams momentum, 
                                        const TypeKey *sample_id,
                                        const TypeValueIndex *hash_value_index_sort, 
                                        const uint32_t *hash_value_index_count,
                                        const uint32_t *hash_value_index_count_offset, 
                                        const float *wgrad,
                                        TypeValueIndex *deltaw_hash_value_index, 
                                        float *deltaw) {
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
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
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

// calculate weights update value(deltaw) by nesterov opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_nesterov_kernel(const uint32_t hash_value_index_count_num, 
                                    const int embedding_vec_size, 
                                    const float lr,
                                    const NesterovOptHyperParams nesterov, 
                                    const TypeKey *sample_id,
                                    const TypeValueIndex *hash_value_index_sort, 
                                    const uint32_t *hash_value_index_count,
                                    const uint32_t *hash_value_index_count_offset, 
                                    const float *wgrad,
                                    TypeValueIndex *deltaw_hash_value_index, 
                                    float *deltaw) {
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
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
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

// update embedding table(weights) by deltaw
template <typename TypeValueIndex>
__global__ void update_kernel(const uint32_t hash_value_index_count_num,
                              const int embedding_vec_size,
                              const TypeValueIndex *deltaw_hash_value_index, 
                              const float *deltaw,
                              float *hash_table_value) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if ((bid < hash_value_index_count_num) && (tid < embedding_vec_size)) {
    TypeValueIndex value_index = deltaw_hash_value_index[bid];
    long long feature_index = value_index * embedding_vec_size + tid;

    hash_table_value[feature_index] += deltaw[bid * embedding_vec_size + tid];
  }
}

// memset liner data to the buffer
template <typename Type>
__global__ void memset_liner(Type *data, 
                             const Type start_value, 
                             const Type stride_value, 
                             const long long n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = start_value + (Type)gid * stride_value;
  }
}

template <typename TypeValueIndex>
__global__ void get_hash_table_value(const long long count, 
                                    const int embedding_vec_size,
                                    const TypeValueIndex *value_index,
                                    const float *hash_table_value, 
                                    float *value_retrieved) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < count && tid < embedding_vec_size) {
    TypeValueIndex index = value_index[bid];
    value_retrieved[bid * embedding_vec_size + tid] =
        hash_table_value[index * embedding_vec_size + tid];
  }
}

}  // end of namespace HugeCTR
