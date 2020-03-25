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

#pragma once

#include "HugeCTR/include/common.hpp"

//#include <cooperative_groups.h>
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
__global__ void forward_sum_kernel(int batch_size, 
                                   int slot_num,
                                   int embedding_vec_size, 
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
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] = sum;
    }
  }
}

// forward kernel function: this is an additional function for combiner=mean
template <typename TypeKey>
__global__ void forward_scale_kernel(int batch_size, 
                                     int slot_num,
                                     int embedding_vec_size, 
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

// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeKey, typename TypeValueIndex>
__global__ void forward_mean_kernel(int batch_size, 
                                   int slot_num,
                                   int embedding_vec_size, 
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
      }

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / (float)feature_num;
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] = (sum * scaler);
    }
  }
}

// backward kernel function: for combiner=sum
template <typename TypeKey>
__global__ void backward_sum_kernel(int batch_size, 
                                    int slot_num,
                                    int embedding_vec_size, 
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
__global__ void backward_mean_kernel(int batch_size, 
                                     int slot_num,
                                     int embedding_vec_size, 
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
__global__ void sample_id_expand_kernel(int batch_size, 
                                        int slot_num,
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



template <typename TypeValueIndex>
__global__ void value_count_kernel_1(int nnz, 
				     const TypeValueIndex *hash_value_index_sort,
				     uint32_t *new_hash_value_flag)
				     // uint32_t *hash_value_index_count,
				     // uint32_t *hash_value_index_count_offset,
				     // uint32_t *hash_value_index_count_counter) {
{  
  for(int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x*gridDim.x){
    TypeValueIndex cur_value = hash_value_index_sort[gid];
    if (gid > 0) {
      TypeValueIndex former_value = hash_value_index_sort[gid - 1];
      // decide if this is the start of a group(the elements in this group have the same
      // hash_value_index_sort)
      if (cur_value != former_value) {
        new_hash_value_flag[gid] = 1;
      } else {
        new_hash_value_flag[gid] = 0;
      }
    } else {  // gid == 0
      new_hash_value_flag[gid] = 1;
    }
  }
}

__global__ void value_count_kernel_2(int nnz, 
				     const uint32_t *new_hash_value_flag,
				     const uint32_t *hash_value_flag_sumed,
				     uint32_t *hash_value_index_index,
				     uint32_t *counter)

{ 
  for(int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x*gridDim.x){
    uint32_t flag = new_hash_value_flag[gid];
    if(flag == 1){
	hash_value_index_index[hash_value_flag_sumed[gid]-1] = gid;
    }
  }
  if(blockIdx.x * blockDim.x + threadIdx.x == 0){
    *counter = hash_value_flag_sumed[nnz-1];
    hash_value_index_index[*counter] = nnz;
  }
}

__global__ void adam_update_kernel_global( const int embedding_vec_size,
				    const int table_size, //vocabulary size / factor
				    const AdamOptHyperParams adam,
				    float *hash_table_value) {
  const int TILE_SIZE = blockDim.x*gridDim.x;
  for(  int feature_index = blockIdx.x*blockDim.x + threadIdx.x; 
	feature_index < table_size*embedding_vec_size; feature_index += TILE_SIZE){
    float mi = adam.m_ptr[feature_index];
    float vi = adam.v_ptr[feature_index];
    float weight_diff = -adam.alpha_t * mi / (sqrtf(vi) + adam.epsilon);
    hash_table_value[feature_index] += weight_diff;
    adam.m_ptr[feature_index] = adam.beta1 * mi;
    adam.v_ptr[feature_index] = adam.beta2 * vi;
  }
}

// calculate weights update value(deltaw) by adam opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_adam_kernel_global(const uint32_t hash_value_index_count_num,
				       const int embedding_vec_size, 
				       const AdamOptHyperParams adam,
				       const TypeKey *sample_id,
				       const TypeValueIndex *hash_value_index_sort,
				       const uint32_t *hash_value_index_count_offset, 
				       const float *wgrad, 
				       const float scaler) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
    float mi = adam.m_ptr[feature_index] + (1.0f - adam.beta1) * gi;
    float vi = adam.v_ptr[feature_index] + (1.0f - adam.beta2) * gi * gi;
    adam.m_ptr[feature_index] = mi;
    adam.v_ptr[feature_index] = vi;
  }
}

// calculate weights update value(deltaw) by momentum_sgd opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_momentum_sgd_kernel_global(uint32_t hash_value_index_count_num, 
					       int embedding_vec_size, 
					       float lr,
					       const MomentumSgdOptHyperParams momentum, 
					       const TypeKey *sample_id,
					       const TypeValueIndex *hash_value_index_sort, 
					       const uint32_t *hash_value_index_count_offset,
					       const float *wgrad,
					       const float scaler) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];
    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }

    gi = gi / scaler;
    // compute the grad of the weights and update it
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
    float mo = momentum.momentum_ptr[feature_index] - lr * gi;
    momentum.momentum_ptr[feature_index] = mo;
  }
}

__global__ void momentum_sgd_update_kernel_global( const int embedding_vec_size,
						   const int table_size, //vocabulary size / factor
						   const MomentumSgdOptHyperParams momentum, 
						   float *hash_table_value) {
  const int TILE_SIZE = blockDim.x*gridDim.x;
  for(  int feature_index = blockIdx.x*blockDim.x + threadIdx.x; 
	feature_index < table_size*embedding_vec_size; feature_index += TILE_SIZE){
    float mo = momentum.momentum_ptr[feature_index];
    hash_table_value[feature_index] += mo;
    momentum.momentum_ptr[feature_index] = mo*momentum.factor;
  }
}


__global__ void nesterov_global_update_kernel_global( const int embedding_vec_size,
					       const int table_size, //vocabulary size / factor
					       const NesterovOptHyperParams nesterov, 
					       float *hash_table_value) {
  const int TILE_SIZE = blockDim.x*gridDim.x;
  for(  int feature_index = blockIdx.x*blockDim.x + threadIdx.x; 
	feature_index < table_size*embedding_vec_size; feature_index += TILE_SIZE){
    nesterov.accm_ptr[feature_index]*=nesterov.mu;
    float accm = nesterov.accm_ptr[feature_index];
    hash_table_value[feature_index] += accm*nesterov.mu;
  }
}


// calculate weights update value(deltaw) by nesterov opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void nesterov_local_update_kernel_global(uint32_t hash_value_index_count_num, 
						    int embedding_vec_size, 
						    float lr,
						    const NesterovOptHyperParams nesterov, 
						    const TypeKey *sample_id,
						    const TypeValueIndex *hash_value_index_sort, 
						    const uint32_t *hash_value_index_count_offset, 
						    const float *wgrad,
						    float *hash_table_value,
						    const float scaler) {
  

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    TypeValueIndex row_index = hash_value_index_sort[offset];
    TypeValueIndex feature_index = row_index * embedding_vec_size + tid;
    nesterov.accm_ptr[feature_index] -= lr * gi;
    hash_table_value[feature_index] -= (1+nesterov.mu)*(lr * gi);
  }
}


// calculate weights update value(deltaw) by adam opitimizer
template <typename TypeKey, typename TypeValueIndex>
__global__ void opt_adam_kernel(const uint32_t hash_value_index_count_num,
                                const int embedding_vec_size, 
                                const AdamOptHyperParams adam,
                                const TypeKey *sample_id,
                                const TypeValueIndex *hash_value_index_sort,
                                const uint32_t *hash_value_index_count_offset, 
                                const float *wgrad,
                                TypeValueIndex *deltaw_hash_value_index, 
                                float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }
    gi = gi / scaler;
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
                                        const uint32_t *hash_value_index_count_offset, 
                                        const float *wgrad,
                                        TypeValueIndex *deltaw_hash_value_index, 
                                        float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];
    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }
    gi = gi / scaler;
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
                                    const uint32_t *hash_value_index_count_offset, 
                                    const float *wgrad,
                                    TypeValueIndex *deltaw_hash_value_index, 
                                    float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num = hash_value_index_count_offset[bid+1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += wgrad[sample_index * embedding_vec_size + tid];
    }
    gi = gi / scaler;
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
__global__ void memset_liner_kernel(Type *data, 
                             const Type start_value, 
                             const Type stride_value, 
                             const long long n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < n) {
    data[gid] = start_value + (Type)gid * stride_value;
  }
}

// get hash_value by value_index from hash_table_value matrix
template <typename TypeValueIndex>
__global__ void get_hash_value_kernel(long long count, 
                                      int embedding_vec_size,
                                      const TypeValueIndex *value_index,
                                      const float *hash_table_value, 
                                      float *value_retrieved) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid < count && tid < embedding_vec_size) {
    TypeValueIndex index = value_index[bid]; // row number in the hash_table_value matrix
    value_retrieved[bid * embedding_vec_size + tid] =
        hash_table_value[index * embedding_vec_size + tid];
  }
}

// get slot_id from hash_table_slot_id vector by value_index
template<typename TypeValueIndex>
__global__ void get_hash_slot_id_kernel(int count, 
                                        const TypeValueIndex * value_index,
                                        const TypeValueIndex * hash_table_slot_id, 
                                        TypeValueIndex * slot_id) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < count) {
    TypeValueIndex index = value_index[gid];
    slot_id[gid] = hash_table_slot_id[index];
  }
}

// reorder operation after all2all in forward propagation 
template <typename Type>
__global__ void forward_reorder_kernel(int batch_size_per_gpu,
                                      int slot_num,
                                      int embedding_vec_size,
                                      int gpu_num,
                                      const Type * input,
                                      Type * output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding vector
  // gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one sample on each GPU
  // Each thread needs to process slot_num slots

  int tid  = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid; // sample_id on the current GPU 

  if((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    int dst_offset = sample_id * slot_num * embedding_vec_size; // offset for the first slot of one sample
    int dst_stride = embedding_vec_size; // stride from slot to slot

    for(int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num; 
      int offset_pre = 0; // offset in previous gpus
      for(int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id<(slot_num % gpu_num))? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;
      }
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id<(slot_num % gpu_num))? 1 : 0);
      int offset_cur = sample_id * slot_num_per_gpu; // offset in current gpu 
      int src_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int dst_addr = dst_offset + dst_stride * slot_id;
      output[dst_addr+tid] = input[src_addr+tid];

      // // just for debug 
      // printf("bid=%d, tid=%d, slot_id=%d, src_addr=%d, dst_addr=%d, input=%f, output=%f\n",\
      // bid, tid, slot_id, src_addr, dst_addr, input[src_addr+tid], output[dst_addr+tid]);
    }
  }
}

// reorder operation before all2all in backward propagation 
template <typename Type>
__global__ void backward_reorder_kernel(int batch_size_per_gpu,
                                        int slot_num,
                                        int embedding_vec_size,
                                        int gpu_num,
                                        const Type * input,
                                        Type * output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding vector
  // gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one sample on each GPU
  // Each thread needs to process slot_num slots

  int tid  = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid; // sample_id on the current GPU 

  if((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    int src_offset = sample_id * slot_num * embedding_vec_size; 
    int src_stride = embedding_vec_size; 

    for(int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num; 
      int offset_pre = 0; // offset in previous gpus
      for(int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id<(slot_num % gpu_num))? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;
      }
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id<(slot_num % gpu_num))? 1 : 0);
      int offset_cur = sample_id * slot_num_per_gpu; // offset in current gpu 
      int dst_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int src_addr =  src_offset + src_stride * slot_id;
      output[dst_addr+tid] = input[src_addr+tid];
    }
  }
}

// store slot_id by row_offset and value_index
template <typename TypeKey, typename TypeValueIndex>
__global__ void store_slot_id_kernel(int batch_size, 
                                    int slot_num, // total slot number in hash table
                                    int slot_num_per_gpu,
                                    int gpu_num, // total gpu number
                                    int gpu_id, // global gpu device id
                                    const TypeKey *row_offset, 
                                    const TypeValueIndex *value_index,
                                    TypeValueIndex *slot_id) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  //int slot_num_per_gpu = (slot_num + gpu_num - 1) / gpu_num;

  if (gid < (batch_size * slot_num_per_gpu)) {
    int sid = gid % slot_num_per_gpu;
    sid = gpu_id + sid * gpu_num; // global slot id 
    if(sid < slot_num) {
      TypeKey offset = row_offset[gid];
      int value_num = row_offset[gid + 1] - offset;
      for (int i = 0; i < value_num; i++) {
        TypeValueIndex index = value_index[offset+i]; // row number 
        // TODO: slot_id may be filled in repeatly 
        slot_id[index] = sid;
      }
    }
  }
}

}  // end of namespace HugeCTR
