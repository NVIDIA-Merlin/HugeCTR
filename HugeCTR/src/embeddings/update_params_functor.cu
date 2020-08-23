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
#include "HugeCTR/include/utils.cuh"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_scan.cuh"

namespace HugeCTR {

namespace {

__global__ void value_count_kernel_2(int nnz, const uint32_t *new_hash_value_flag,
                                     const uint32_t *hash_value_flag_sumed,
                                     uint32_t *hash_value_index_index, uint32_t *counter)

{
  for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x * gridDim.x) {
    uint32_t flag = new_hash_value_flag[gid];
    if (flag == 1) {
      hash_value_index_index[hash_value_flag_sumed[gid] - 1] = gid;
    }
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    *counter = hash_value_flag_sumed[nnz - 1];
    hash_value_index_index[*counter] = nnz;
  }
}

// expand sample id by row_offset
template <typename TypeKey>
__global__ void sample_id_expand_kernel(int batch_size, int slot_num, const TypeKey *row_offset,
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

__global__ void value_count_kernel_1(int nnz, const size_t *hash_value_index_sort,
                                     uint32_t *new_hash_value_flag) {
  for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x * gridDim.x) {
    size_t cur_value = hash_value_index_sort[gid];
    if (gid > 0) {
      size_t former_value = hash_value_index_sort[gid - 1];
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

// calculate weights update value(deltaw) by adam opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adam_kernel_global(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                       const AdamOptHyperParams<TypeEmbeddingComp> adam,
                                       const TypeKey *sample_id,
                                       const size_t *hash_value_index_sort,
                                       const uint32_t *hash_value_index_count_offset,
                                       const TypeEmbeddingComp *wgrad, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    // uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.m_ptr[feature_index]) +
               (1.0f - adam.beta1) * gi / adam.beta1;

    float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.v_ptr[feature_index]) +
               (1.0f - adam.beta2) * gi * gi / adam.beta2;

    adam.m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    adam.v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
  }
}

template <typename TypeEmbeddingComp>
__global__ void adam_update_kernel_global(int embedding_vec_size,
                                          size_t table_size,  // vocabulary size / factor
                                          const AdamOptHyperParams<TypeEmbeddingComp> adam,
                                          float alpha_t, float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float mi =
        adam.beta1 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.m_ptr[feature_index]);
    float vi =
        adam.beta2 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.v_ptr[feature_index]);

    adam.m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    adam.v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);

    float weight_diff = -alpha_t * mi / (sqrtf(vi) + adam.epsilon);
    hash_table_value[feature_index] += weight_diff;
  }
}

// calculate weights update value(deltaw) by momentum_sgd opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_momentum_sgd_kernel_global(
    uint32_t hash_value_index_count_num, int embedding_vec_size, float lr,
    const MomentumSGDOptHyperParams<TypeEmbeddingComp> momentum, const TypeKey *sample_id,
    const size_t *hash_value_index_sort, const uint32_t *hash_value_index_count_offset,
    const TypeEmbeddingComp *wgrad, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];
    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }

    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mo =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum.momentum_ptr[feature_index]) -
        lr * gi / momentum.factor;
    momentum.momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
  }
}

template <typename TypeEmbeddingComp>
__global__ void momentum_sgd_update_kernel_global(
    int embedding_vec_size,
    size_t table_size,  // vocabulary size / factor
    const MomentumSGDOptHyperParams<TypeEmbeddingComp> momentum, float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float mo =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum.momentum_ptr[feature_index]);
    mo *= momentum.factor;
    hash_table_value[feature_index] += mo;
    momentum.momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
  }
}

template <typename TypeEmbeddingComp>
__global__ void nesterov_global_update_kernel_global(
    int embedding_vec_size,
    size_t table_size,  // vocabulary size / factor
    const NesterovOptHyperParams<TypeEmbeddingComp> nesterov, float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float accm =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(nesterov.accm_ptr[feature_index]);
    accm *= nesterov.mu;
    nesterov.accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);
    hash_table_value[feature_index] += accm * nesterov.mu;
  }
}

// calculate weights update value(deltaw) by nesterov opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void nesterov_local_update_kernel_global(
    uint32_t hash_value_index_count_num, int embedding_vec_size, float lr,
    const NesterovOptHyperParams<TypeEmbeddingComp> nesterov, const TypeKey *sample_id,
    const size_t *hash_value_index_sort, const uint32_t *hash_value_index_count_offset,
    const TypeEmbeddingComp *wgrad, float *hash_table_value, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float accm =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(nesterov.accm_ptr[feature_index]);
    accm -= lr * gi;
    nesterov.accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);
    hash_table_value[feature_index] -= (1 + nesterov.mu) * (lr * gi);
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_sgd_kernel_global(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                      float lr, const TypeKey *sample_id,
                                      const size_t *hash_value_index_sort,
                                      const uint32_t *hash_value_index_count_offset,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                      float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    // uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;

    // update
    size_t value_index = hash_value_index_sort[offset];
    size_t feature_index = value_index * embedding_vec_size + tid;
    hash_table_value[feature_index] -= lr * gi;
  }
}

// calculate weights update value(deltaw) by adam opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adam_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                const AdamOptHyperParams<TypeEmbeddingComp> adam, float alpha_t,
                                const TypeKey *sample_id, const size_t *hash_value_index_sort,
                                const uint32_t *hash_value_index_count_offset,
                                const TypeEmbeddingComp *wgrad, size_t *deltaw_hash_value_index,
                                float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    // uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mi =
        adam.beta1 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.m_ptr[feature_index]) +
        (1.0f - adam.beta1) * gi;
    float vi =
        adam.beta2 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(adam.v_ptr[feature_index]) +
        (1.0f - adam.beta2) * gi * gi;
    adam.m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    adam.v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
    float weight_diff = -alpha_t * mi / (sqrtf(vi) + adam.epsilon);

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = weight_diff;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

// calculate weights update value(deltaw) by momentum_sgd opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_momentum_sgd_kernel(
    uint32_t hash_value_index_count_num, int embedding_vec_size, float lr,
    const MomentumSGDOptHyperParams<TypeEmbeddingComp> momentum, const TypeKey *sample_id,
    const size_t *hash_value_index_sort, const uint32_t *hash_value_index_count_offset,
    const TypeEmbeddingComp *wgrad, size_t *deltaw_hash_value_index, float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];
    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mo = momentum.factor * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                     momentum.momentum_ptr[feature_index]) -
               lr * gi;
    momentum.momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = mo;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

// calculate weights update value(deltaw) by nesterov opitimizer
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_nesterov_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                    float lr,
                                    const NesterovOptHyperParams<TypeEmbeddingComp> nesterov,
                                    const TypeKey *sample_id, const size_t *hash_value_index_sort,
                                    const uint32_t *hash_value_index_count_offset,
                                    const TypeEmbeddingComp *wgrad, size_t *deltaw_hash_value_index,
                                    float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    //    uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;
    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float accm_old =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(nesterov.accm_ptr[feature_index]);
    float accm_new = nesterov.mu * accm_old - lr * gi;
    nesterov.accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm_new);
    float weight_diff = -nesterov.mu * accm_old + (1.0f + nesterov.mu) * accm_new;

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = weight_diff;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_sgd_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                               float lr, const TypeKey *sample_id,
                               const size_t *hash_value_index_sort,
                               const uint32_t *hash_value_index_count_offset,
                               const TypeEmbeddingComp *wgrad, size_t *deltaw_hash_value_index,
                               float *deltaw, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    // uint32_t sample_num = hash_value_index_count[bid];
    uint32_t sample_num =
        hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

    // accumulate the wgrads for the corresponding embedding vector
    float gi = 0.0f;
    uint32_t offset = hash_value_index_count_offset[bid];
    for (int i = 0; i < sample_num; i++) {
      int sample_index = sample_id[offset + i];
      gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          wgrad[sample_index * embedding_vec_size + tid]);
    }
    gi = gi / scaler;

    // compute the grad of the weights and update it
    size_t row_index = hash_value_index_sort[offset];
    float weight_diff = -lr * gi;

    // save weights diff
    deltaw[bid * embedding_vec_size + tid] = weight_diff;

    // save hash value_indexs(corresponding to deltaw)
    if (tid == 0) {
      deltaw_hash_value_index[bid] = row_index;
    }
  }
}

// update embedding table(weights) by deltaw
__global__ void update_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                              const size_t *deltaw_hash_value_index, const float *deltaw,
                              float *hash_table_value) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if ((bid < hash_value_index_count_num) && (tid < embedding_vec_size)) {
    size_t value_index = deltaw_hash_value_index[bid];
    size_t feature_index = value_index * embedding_vec_size + tid;
    hash_table_value[feature_index] += deltaw[bid * embedding_vec_size + tid];
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_sgd_atomic_kernel(int nnz, int embedding_vec_size, float lr_scale,
                                      const size_t *hash_value_index, const TypeKey *sample_ids,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < nnz) {
    for (int key_id = bid; key_id < nnz; key_id += gridDim.x) {
      int sample_id = sample_ids[key_id];
      float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                     wgrad[sample_id * embedding_vec_size + tid]);

      // atomic update
      size_t value_index = hash_value_index[key_id];
      size_t feature_index = value_index * embedding_vec_size + tid;
      atomicAdd(&hash_table_value[feature_index], deltaw);
    }
  }
}

// only support LocalizedSlotSparseEmbeddingOneHot
template <typename TypeEmbeddingComp>
__global__ void opt_sgd_atomic_kernel(int nnz, int embedding_vec_size, float lr_scale,
                                      const size_t *hash_value_index,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < nnz) {
    for (int key_id = bid; key_id < nnz; key_id += gridDim.x) {
      // for one-hot, the max_feature_per_slot is 1, so sample_id is equal to key_id
      float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                     wgrad[key_id * embedding_vec_size + tid]);

      // atomic update
      size_t value_index = hash_value_index[key_id];
      size_t feature_index = value_index * embedding_vec_size + tid;
      atomicAdd(&hash_table_value[feature_index], deltaw);
    }
  }
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::update_params(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, OptParams<TypeEmbeddingComp> &opt_params, size_t nnz,
    const Tensor2<TypeHashKey> &row_offset, Tensor2<size_t> &hash_value_index,
    Tensor2<TypeHashKey> &sample_id, Tensor2<TypeHashKey> &sample_id_sort,
    Tensor2<size_t> &hash_value_index_sort, Tensor2<uint32_t> &hash_value_index_count_offset,
    Tensor2<uint32_t> &new_hash_value_flag, Tensor2<uint32_t> &hash_value_flag_sumed,
    Tensor2<uint32_t> &hash_value_index_count_counter, Tensor2<void> &temp_storage_sort,
    Tensor2<void> &temp_storage_scan, const Tensor2<TypeEmbeddingComp> &wgrad,
    Tensor2<size_t> &deltaw_hash_value_index, Tensor2<float> &deltaw,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream) {
  if (slot_num == 0) {
    return;
  }

  size_t block_size, grid_size;

  try {
    // step1: expand sample IDs
    block_size = 64;
    grid_size = (batch_size * slot_num - 1) / block_size + 1;
    sample_id_expand_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, row_offset.get_ptr(), sample_id.get_ptr());

    if (opt_params.optimizer == Optimizer_t::SGD &&
        opt_params.hyperparams.sgd.atomic_update) {  // for SGD, do atomic update
      const size_t block_size = embedding_vec_size;
      const size_t grid_size = min(max(1ul, nnz), sm_count * 32);

      float lr_scale = opt_params.lr / opt_params.scaler;
      opt_sgd_atomic_kernel<<<grid_size, block_size, 0, stream>>>(
          nnz, embedding_vec_size, lr_scale, hash_value_index.get_ptr(), sample_id.get_ptr(),
          wgrad.get_ptr(), hash_table_value.get_ptr());
    } else {
      // step3: sort by hash_value_index
      int end_bit = static_cast<int>(log2(static_cast<float>(max_vocabulary_size_per_gpu))) + 1;
      size_t temp_storage_sort_size = temp_storage_sort.get_size_in_bytes();
      CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
          temp_storage_sort.get_ptr(), temp_storage_sort_size, hash_value_index.get_ptr(),
          hash_value_index_sort.get_ptr(), sample_id.get_ptr(), sample_id_sort.get_ptr(), nnz, 0,
          end_bit, stream, false));

      // step4: count the number for each unduplicated hash_value_index
      CK_CUDA_THROW_(
          cudaMemsetAsync(hash_value_index_count_counter.get_ptr(), 0, sizeof(uint32_t), stream));

      constexpr size_t max_grid_size = 384;
      block_size = 256;
      grid_size = min(max_grid_size, (nnz - 1) / block_size + 1);

      value_count_kernel_1<<<grid_size, block_size, 0, stream>>>(
          nnz, hash_value_index_sort.get_ptr(), new_hash_value_flag.get_ptr());

      // prefix_sum
      size_t temp_storage_scan_size = temp_storage_scan.get_size_in_bytes();
      CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
          temp_storage_scan.get_ptr(), temp_storage_scan_size, new_hash_value_flag.get_ptr(),
          hash_value_flag_sumed.get_ptr(), nnz, stream));

      value_count_kernel_2<<<grid_size, block_size, 0, stream>>>(
          nnz, new_hash_value_flag.get_ptr(), hash_value_flag_sumed.get_ptr(),
          hash_value_index_count_offset.get_ptr(), hash_value_index_count_counter.get_ptr());

      uint32_t hash_hash_value_index_count_num = 0;
      // this async memcpy will not perform as a async operation because the host memory is not
      // a pinned memroy
      CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num,
                                     hash_value_index_count_counter.get_ptr(), sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, stream));

      // step5: use optimizer method to compute deltaw, and record corresponding
      // deltaw_hash_value_index
      block_size = embedding_vec_size;
      grid_size = max(1, hash_hash_value_index_count_num);

      if (opt_params.global_update) {
        switch (opt_params.optimizer) {
          case Optimizer_t::Adam:  // adam
          {
            float alpha_t =
                opt_params.lr *
                sqrt(1 -
                     pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));
            // update target mi and vi
            opt_adam_kernel_global<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(), opt_params.scaler);
            // all update according to the mi vi
            adam_update_kernel_global<<<1024, 256, 0, stream>>>(
                embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.adam,
                alpha_t, hash_table_value.get_ptr());
            break;
          }
          case Optimizer_t::MomentumSGD:  // momentum sgd
            opt_momentum_sgd_kernel_global<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                opt_params.hyperparams.momentum, sample_id_sort.get_ptr(),
                hash_value_index_sort.get_ptr(), hash_value_index_count_offset.get_ptr(),
                wgrad.get_ptr(), opt_params.scaler);
            momentum_sgd_update_kernel_global<<<1024, 256, 0, stream>>>(
                embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.momentum,
                hash_table_value.get_ptr());
            break;
          case Optimizer_t::Nesterov:  // nesterov
            nesterov_global_update_kernel_global<<<1024, 256, 0, stream>>>(
                embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.nesterov,
                hash_table_value.get_ptr());
            nesterov_local_update_kernel_global<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                opt_params.hyperparams.nesterov, sample_id_sort.get_ptr(),
                hash_value_index_sort.get_ptr(), hash_value_index_count_offset.get_ptr(),
                wgrad.get_ptr(), hash_table_value.get_ptr(), opt_params.scaler);
            break;
          case Optimizer_t::SGD:
            opt_sgd_kernel_global<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                hash_table_value.get_ptr(), opt_params.scaler);
            break;
          default:
            CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
        }
      } else {
        switch (opt_params.optimizer) {
          case Optimizer_t::Adam:  // adam
          {
            float alpha_t =
                opt_params.lr *
                sqrt(1 -
                     pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

            opt_adam_kernel<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                alpha_t, sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                deltaw_hash_value_index.get_ptr(), deltaw.get_ptr(), opt_params.scaler);
            break;
          }
          case Optimizer_t::MomentumSGD:  // momentum sgd
            opt_momentum_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                opt_params.hyperparams.momentum, sample_id_sort.get_ptr(),
                hash_value_index_sort.get_ptr(), hash_value_index_count_offset.get_ptr(),
                wgrad.get_ptr(), deltaw_hash_value_index.get_ptr(), deltaw.get_ptr(),
                opt_params.scaler);
            break;
          case Optimizer_t::Nesterov:  // nesterov
            opt_nesterov_kernel<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                opt_params.hyperparams.nesterov, sample_id_sort.get_ptr(),
                hash_value_index_sort.get_ptr(), hash_value_index_count_offset.get_ptr(),
                wgrad.get_ptr(), deltaw_hash_value_index.get_ptr(), deltaw.get_ptr(),
                opt_params.scaler);
            break;
          case Optimizer_t::SGD:
            opt_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                deltaw_hash_value_index.get_ptr(), deltaw.get_ptr(), opt_params.scaler);
            break;
          default:
            CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
        }

        // step6: update hash_table_value by deltaw
        block_size = embedding_vec_size;
        grid_size = max(1, hash_hash_value_index_count_num);
        update_kernel<<<grid_size, block_size, 0, stream>>>(
            hash_hash_value_index_count_num, embedding_vec_size, deltaw_hash_value_index.get_ptr(),
            deltaw.get_ptr(), hash_table_value.get_ptr());
      }  // else

    }  // else
#ifndef NDEBUG
    cudaDeviceSynchronize();
    CK_CUDA_THROW_(cudaGetLastError());
#endif
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::update_params(size_t embedding_vec_size,
                                            const OptParams<TypeEmbeddingComp> &opt_params,
                                            size_t nnz, const Tensor2<size_t> &hash_value_index,
                                            const Tensor2<TypeEmbeddingComp> &wgrad,
                                            Tensor2<float> &hash_table_value, size_t sm_count,
                                            cudaStream_t stream) {
  try {
    if (opt_params.optimizer == Optimizer_t::SGD && opt_params.hyperparams.sgd.atomic_update) {
      const size_t grid_size = min(max(1ul, nnz), sm_count * 32);
      const size_t block_size = embedding_vec_size;

      float lr_scale = opt_params.lr / opt_params.scaler;

      // for one-hot, the sample_id is dedicated.
      opt_sgd_atomic_kernel<<<grid_size, block_size, 0, stream>>>(
          nnz, embedding_vec_size, lr_scale, hash_value_index.get_ptr(), wgrad.get_ptr(),
          hash_table_value.get_ptr());
    } else {
      CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
    }

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::update_params<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, OptParams<float> &opt_params, size_t nnz,
    const Tensor2<unsigned int> &row_offset, Tensor2<size_t> &hash_value_index,
    Tensor2<unsigned int> &sample_id, Tensor2<unsigned int> &sample_id_sort,
    Tensor2<size_t> &hash_value_index_sort, Tensor2<uint32_t> &hash_value_index_count_offset,
    Tensor2<uint32_t> &new_hash_value_flag, Tensor2<uint32_t> &hash_value_flag_sumed,
    Tensor2<uint32_t> &hash_value_index_count_counter, Tensor2<void> &temp_storage_sort,
    Tensor2<void> &temp_storage_scan, const Tensor2<float> &wgrad,
    Tensor2<size_t> &deltaw_hash_value_index, Tensor2<float> &deltaw,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, OptParams<float> &opt_params, size_t nnz,
    const Tensor2<long long> &row_offset, Tensor2<size_t> &hash_value_index,
    Tensor2<long long> &sample_id, Tensor2<long long> &sample_id_sort,
    Tensor2<size_t> &hash_value_index_sort, Tensor2<uint32_t> &hash_value_index_count_offset,
    Tensor2<uint32_t> &new_hash_value_flag, Tensor2<uint32_t> &hash_value_flag_sumed,
    Tensor2<uint32_t> &hash_value_index_count_counter, Tensor2<void> &temp_storage_sort,
    Tensor2<void> &temp_storage_scan, const Tensor2<float> &wgrad,
    Tensor2<size_t> &deltaw_hash_value_index, Tensor2<float> &deltaw,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, OptParams<__half> &opt_params, size_t nnz,
    const Tensor2<unsigned int> &row_offset, Tensor2<size_t> &hash_value_index,
    Tensor2<unsigned int> &sample_id, Tensor2<unsigned int> &sample_id_sort,
    Tensor2<size_t> &hash_value_index_sort, Tensor2<uint32_t> &hash_value_index_count_offset,
    Tensor2<uint32_t> &new_hash_value_flag, Tensor2<uint32_t> &hash_value_flag_sumed,
    Tensor2<uint32_t> &hash_value_index_count_counter, Tensor2<void> &temp_storage_sort,
    Tensor2<void> &temp_storage_scan, const Tensor2<__half> &wgrad,
    Tensor2<size_t> &deltaw_hash_value_index, Tensor2<float> &deltaw,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, OptParams<__half> &opt_params, size_t nnz,
    const Tensor2<long long> &row_offset, Tensor2<size_t> &hash_value_index,
    Tensor2<long long> &sample_id, Tensor2<long long> &sample_id_sort,
    Tensor2<size_t> &hash_value_index_sort, Tensor2<uint32_t> &hash_value_index_count_offset,
    Tensor2<uint32_t> &new_hash_value_flag, Tensor2<uint32_t> &hash_value_flag_sumed,
    Tensor2<uint32_t> &hash_value_index_count_counter, Tensor2<void> &temp_storage_sort,
    Tensor2<void> &temp_storage_scan, const Tensor2<__half> &wgrad,
    Tensor2<size_t> &deltaw_hash_value_index, Tensor2<float> &deltaw,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<float>(
    size_t embedding_vec_size, const OptParams<float> &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<float> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<__half>(
    size_t embedding_vec_size, const OptParams<__half> &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<__half> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

}  // namespace HugeCTR
