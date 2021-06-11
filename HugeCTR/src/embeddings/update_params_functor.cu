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
#include <thrust/sort.h>  // for implictly including cub headers

namespace HugeCTR {

namespace {

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
void SparseEmbeddingFunctors::update_params(size_t embedding_vec_size,
                                            const OptParams &opt_params,
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

template void SparseEmbeddingFunctors::update_params<float>(
    size_t embedding_vec_size, const OptParams &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<float> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

template void SparseEmbeddingFunctors::update_params<__half>(
    size_t embedding_vec_size, const OptParams &opt_params, size_t nnz,
    const Tensor2<size_t> &hash_value_index, const Tensor2<__half> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream);

}  // namespace HugeCTR
