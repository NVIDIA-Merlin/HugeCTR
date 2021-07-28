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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
size_t EmbeddingTableFunctors<dtype>::get_embedding_table_index(
    const std::vector<size_t>& table_sizes, dtype category) {
  size_t embedding = 0;
  dtype next_offset = (dtype)table_sizes[embedding];
  for (embedding = 0; embedding < table_sizes.size() - 1 && category >= next_offset; ++embedding)
    next_offset += table_sizes[embedding + 1];
  return embedding;
}

template <typename dtype>
void EmbeddingTableFunctors<dtype>::get_embedding_offsets(std::vector<dtype>& embedding_offsets,
                                                          const std::vector<size_t>& table_sizes) {
  const size_t num_tables = table_sizes.size();
  embedding_offsets.resize(num_tables);
  dtype embedding_offset = (dtype)0;
  for (size_t embedding = 0; embedding < num_tables; ++embedding) {
    embedding_offsets[embedding] = embedding_offset;
    embedding_offset += table_sizes[embedding];
  }
}

template <typename dtype>
dtype EmbeddingTableFunctors<dtype>::get_num_categories(const std::vector<size_t>& table_sizes) {
  dtype num_categories = (dtype)0;
  for (size_t i = 0; i < table_sizes.size(); ++i) num_categories += table_sizes[i];
  return num_categories;
}

template <typename dtype>
__global__ void data_to_unique_categories_kernel(const dtype* __restrict__ data,
                                                 const dtype* __restrict__ embedding_offsets,
                                                 int num_tables, int num_data,
                                                 dtype* __restrict__ samples) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_data;
       idx += blockDim.x * gridDim.x) {
    samples[idx] = data[idx] + embedding_offsets[idx % num_tables];
  }
}

template <typename dtype>
__global__ void data_to_unique_categories_align2_kernel(dtype* __restrict__ data,
                                                        dtype* __restrict__ embedding_offsets,
                                                        int num_tables, int num_data,
                                                        dtype* __restrict__ samples) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_data;
       idx += blockDim.x * gridDim.x) {
    uint2 load_data = reinterpret_cast<uint2*>(data)[idx];
    uint2 load_embedding_offsets = reinterpret_cast<uint2*>(embedding_offsets)[idx % num_tables];

    load_data.x += load_embedding_offsets.x;
    load_data.y += load_embedding_offsets.y;
    reinterpret_cast<uint2*>(samples)[idx] = load_data;
  }
}

/// data_to_unique_categories converts the argument 'data' and stores
///        the result in member variable 'samples'.
///        Per network, the columns corresponding to embedding tables
///        are concatenated and categories get an unique index / label.
template <typename dtype>
void Data<dtype>::data_to_unique_categories(Tensor2<dtype> data, cudaStream_t stream) {
  /// === TODO: PERFORM ON GPU ===
  /// ============================
  // std::cout << "WARNING: data_to_unique_categories() needs to be placed on the GPU!" <<
  // std::endl;
  // TODO : perform conversion by kernel (before start of iteration ? => see below)
  //        for batch_size = 55*1024
  //        batch_size * 26 * 4 / 1600e9 = 3.67 microseconds,
  //
  // Remark:
  //        Doesn't need to be before start of kernel.
  //        Would be nice to have just before calculating indices, since
  //        those would be in L2 cache already.
  size_t block_size = 256;
  size_t grid_size =
      std::min(static_cast<size_t>(4096),
               (table_sizes.size() * batch_size * num_iterations - 1) / block_size + 1);
  if (table_sizes.size() % 2 == 0 && sizeof(dtype) == 4) {
    data_to_unique_categories_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        data.get_ptr(), embedding_offsets.get_ptr(), table_sizes.size() / 2,
        table_sizes.size() * batch_size * num_iterations / 2, samples.get_ptr());
  } else {
    data_to_unique_categories_kernel<<<grid_size, block_size, 0, stream>>>(
        data.get_ptr(), embedding_offsets.get_ptr(), table_sizes.size(),
        table_sizes.size() * batch_size * num_iterations, samples.get_ptr());
  }
}

template class Data<uint32_t>;
template class Data<long long>;

template struct EmbeddingTableFunctors<uint32_t>;
template struct EmbeddingTableFunctors<long long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
