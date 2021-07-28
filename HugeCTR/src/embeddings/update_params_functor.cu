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

#include <thrust/sort.h>  // for implictly including cub headers

#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.cuh"

#define max_size_top_categories 16
#define num_samples_per_block 128
#define embedding_block_size 128

namespace HugeCTR {

size_t get_max_size_top_categories() { return max_size_top_categories; }
size_t get_num_samples_per_block() { return num_samples_per_block; }
size_t get_embedding_block_size() { return embedding_block_size; }

namespace {

// TODO: it must be moved to SparseOptimizer
// The local memory version of the atomic update kernel - opt_sgd_atomic_kernel for one hot
// embedding.
//
// This function updates the embedding vectors of the top-n features in shared memory
// before writing the accumulated result to global memory. This reduces the number of
// global memory accesses, locks and collisions.
//
// num_samples_per_block number of samples are updated per block and they are iterated over,
// such that all threads update the embedding vector of a single feature simultaneously.
//
// shared ds_top_features_index : the row indices of the top-n - top_features_size - features
// shared ds_embedding : the embedding vector corresponding to the top features (rows)
template <typename TypeEmbeddingComp>
__global__ void opt_sgd_cached_kernel(int nnz, int embedding_vec_size, float lr_scale,
                                      const size_t *top_categories,
                                      const size_t top_categories_size,
                                      const size_t *hash_value_index,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // read a number of top_categories_size top categories indices from global memory
  // note: max_size_top_n (16) less than warp size
  __shared__ size_t ds_top_categories[max_size_top_categories];
  if (tid < top_categories_size) {
    ds_top_categories[tid] = top_categories[tid];
  }
  //__syncthreads();

  // reads num_samples_per_block values indices from hash_value_index into shared memory
  __shared__ size_t ds_category[num_samples_per_block];  // embedding indices for current block
  for (int ds_offset = 0; ds_offset < num_samples_per_block; ds_offset += blockDim.x) {
    int ds_index = ds_offset + tid;
    int key_id = bid * num_samples_per_block + ds_index;
    if (ds_index < num_samples_per_block && key_id < nnz) {
      ds_category[ds_index] = hash_value_index[key_id];
    }
  }
  __syncthreads();

  // map sample category indices to top_category indices
  __shared__ int
      ds_index_top_categories[num_samples_per_block];  // index to top category index array,
                                                       // max_size_top_categories if not present
  {
    for (int ci_offset = 0; ci_offset < num_samples_per_block; ci_offset += blockDim.x) {
      int index_ds_category = ci_offset + tid;
      if (index_ds_category < num_samples_per_block) {
        // loop over top features
        int i_top = max_size_top_categories;  // one past end
        if (index_ds_category + bid * num_samples_per_block < nnz) {
          int category_embedding_index = ds_category[index_ds_category];
          for (int k = 0; k < top_categories_size; ++k) {
            if (category_embedding_index == ds_top_categories[k]) i_top = k;
          }
        }
        ds_index_top_categories[index_ds_category] = i_top;
      }
    }
  }
  __syncthreads();

  // store the sum of deltaw in ds_embedding
  // TODO: make this work for embedding size > 128
  __shared__ float ds_embedding[max_size_top_categories][embedding_block_size];
  // initialize the local embedding vectors
  for (int i = 0; i < top_categories_size; ++i) {
    if (tid < embedding_block_size) {
      ds_embedding[i][tid] = 0.f;
    }
  }
  __syncthreads();

  unsigned int update_top_category = 0;  // bit indicator sequence

  size_t key_id_local = 0;
  for (size_t key_id = bid * num_samples_per_block;
       key_id < nnz && key_id < (bid + 1) * num_samples_per_block; ++key_id) {
    if (tid < embedding_vec_size) {
      int index_top_category = ds_index_top_categories[key_id_local];
      size_t category_embedding_index = ds_category[key_id_local];
      if (index_top_category < max_size_top_categories) {
        // write to shared memory
        update_top_category = (update_top_category | (1 << index_top_category));
        // write results to embedding vector in shared memory
        float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                       wgrad[key_id * embedding_vec_size + tid]);
        ds_embedding[index_top_category][tid] += deltaw;
      } else {
        // write to global memory using atomic
        float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                       wgrad[key_id * embedding_vec_size + tid]);

        // atomic update
        size_t feature_index = category_embedding_index * embedding_vec_size + tid;
        atomicAdd(&hash_table_value[feature_index], deltaw);
      }
    }

    key_id_local++;
  }
  __syncthreads();

  // write the embedding vectors for top features which are in shared memory to global memory
  // for (int i=0; i < max_size_top_categories; ++i) { // maybe this is actually more optimized
  if (tid < embedding_vec_size) {
    for (int i = 0; i < top_categories_size; ++i) {
      // only those that were updated
      if ((update_top_category & (1 << i)) > 0) {
        size_t category_embedding_index = ds_top_categories[i];
        size_t embedding_element_index = category_embedding_index * embedding_vec_size + tid;
        atomicAdd(&hash_table_value[embedding_element_index], ds_embedding[i][tid]);
      }
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

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::opt_sgd_atomic_cached<TypeEmbeddingComp>(
    size_t num_samples, size_t embedding_vec_size, const size_t *hash_value_index, float lr,
    float scaler, const TypeEmbeddingComp *wgrad, float *hash_table_value, size_t *top_categories,
    size_t &size_top_categories, cudaStream_t stream, bool force_stats) {
  static bool perform_stats = true;
  if (perform_stats || force_stats) {
    uint32_t num_unique_categories;
    /// TODO: refactor instead of using placeholder values for the other params
    hybrid_embedding::Statistics<size_t> statistics(num_samples, 1, 1, 1);

    statistics.sort_categories_by_count(hash_value_index, (uint32_t)num_samples, top_categories,
                                        statistics.counts_sorted.get_ptr(), num_unique_categories,
                                        stream);
    size_top_categories = std::min((size_t)num_unique_categories, (size_t)max_size_top_categories);

    perform_stats = false;
  }

  float lr_scale = lr / scaler;
  // treats num_samples_per_block samples
  size_t grid_size = max(1ul, (num_samples - 1) / num_samples_per_block + 1);
  // each thread sets one embedding vector element
  size_t block_size = embedding_vec_size;
  CK_CUDA_THROW_(cudaPeekAtLastError());
  opt_sgd_cached_kernel<<<grid_size, block_size, 0, stream>>>(
      num_samples, embedding_vec_size, lr_scale, top_categories, size_top_categories,
      hash_value_index, wgrad, hash_table_value);
  CK_CUDA_THROW_(cudaPeekAtLastError());
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::update_params<TypeEmbeddingComp>(
    size_t embedding_vec_size, const OptParams &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<TypeEmbeddingComp> &wgrad,
    Tensor2<float> &hash_table_value, Tensor2<size_t> &top_categories, size_t &size_top_categories,
    size_t sm_count, cudaStream_t stream, bool force_stats) {
  try {
    if (opt_params.optimizer == Optimizer_t::SGD && opt_params.hyperparams.sgd.atomic_update) {
      float lr_scale = opt_params.lr / opt_params.scaler;

      opt_sgd_atomic_cached<TypeEmbeddingComp>(nnz, embedding_vec_size, hash_value_index.get_ptr(),
                                               opt_params.lr, opt_params.scaler, wgrad.get_ptr(),
                                               hash_table_value.get_ptr(), top_categories.get_ptr(),
                                               size_top_categories, stream, force_stats);
    } else {
      CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
    }

  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::opt_sgd_atomic_cached<float>(
    size_t num_samples, size_t embedding_vec_size, const size_t *hash_value_index, float lr,
    float scaler, const float *wgrad, float *hash_table_value, size_t *top_categories,
    size_t &size_top_categories, cudaStream_t stream, bool force_stats);

template void SparseEmbeddingFunctors::opt_sgd_atomic_cached<__half>(
    size_t num_samples, size_t embedding_vec_size, const size_t *hash_value_index, float lr,
    float scaler, const __half *wgrad, float *hash_table_value, size_t *top_categories,
    size_t &size_top_categories, cudaStream_t stream, bool force_stats);

template void SparseEmbeddingFunctors::update_params<float>(
    size_t embedding_vec_size, const OptParams &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<float> &wgrad,
    Tensor2<float> &hash_table_value, Tensor2<size_t> &top_categories, size_t &size_top_categories,
    size_t sm_count, cudaStream_t stream, bool force_stats);

template void SparseEmbeddingFunctors::update_params<__half>(
    size_t embedding_vec_size, const OptParams &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<__half> &wgrad,
    Tensor2<float> &hash_table_value, Tensor2<size_t> &top_categories, size_t &size_top_categories,
    size_t sm_count, cudaStream_t stream, bool force_stats);

}  // namespace HugeCTR
