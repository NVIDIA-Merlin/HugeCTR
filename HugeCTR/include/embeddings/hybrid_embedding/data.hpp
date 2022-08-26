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

#pragma once

#include <cuda_runtime.h>

#include <general_buffer2.hpp>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
struct EmbeddingTableFunctors {
  static dtype get_num_categories(const std::vector<size_t> &table_sizes);
  static void get_embedding_offsets(std::vector<dtype> &embedding_offsets,
                                    const std::vector<size_t> &table_sizes);
  static size_t get_embedding_table_index(const std::vector<size_t> &table_sizes, dtype category);
};

// depends on : data reader - or mock data

template <typename dtype>
struct Data {
  std::vector<size_t> table_sizes;
  size_t batch_size;
  size_t num_iterations;
  size_t num_categories;

  Tensor2<dtype> embedding_offsets;
  Tensor2<dtype> samples;

  Data(Tensor2<dtype> samples, const std::vector<size_t> &table_sizes_in, size_t batch_size_in,
       size_t num_iterations_in)
      : samples(samples),
        table_sizes(table_sizes_in),
        batch_size(batch_size_in),
        num_iterations(num_iterations_in) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    buf->reserve({table_sizes_in.size()}, &embedding_offsets);
    buf->allocate();

    std::vector<dtype> h_embedding_offsets;
    EmbeddingTableFunctors<dtype>::get_embedding_offsets(h_embedding_offsets, table_sizes);

    num_categories = EmbeddingTableFunctors<dtype>::get_num_categories(table_sizes);
    HCTR_LIB_THROW(cudaMemcpy(embedding_offsets.get_ptr(), h_embedding_offsets.data(),
                              sizeof(dtype) * h_embedding_offsets.size(), cudaMemcpyHostToDevice));
  }

  Data(const std::vector<size_t> &table_sizes_in, size_t batch_size_in, size_t num_iterations_in)
      : table_sizes(table_sizes_in), batch_size(batch_size_in), num_iterations(num_iterations_in) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    reserve(buf);
    buf->reserve({table_sizes_in.size()}, &embedding_offsets);
    buf->allocate();

    std::vector<dtype> h_embedding_offsets;
    EmbeddingTableFunctors<dtype>::get_embedding_offsets(h_embedding_offsets, table_sizes);

    num_categories = EmbeddingTableFunctors<dtype>::get_num_categories(table_sizes);
    HCTR_LIB_THROW(cudaMemcpy(embedding_offsets.get_ptr(), h_embedding_offsets.data(),
                              sizeof(dtype) * h_embedding_offsets.size(), cudaMemcpyHostToDevice));
  }

  Data() {}
  ~Data() {}

  void reserve(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
    const size_t num_tables = table_sizes.size();
    buf->reserve({num_iterations * batch_size * num_tables, 1}, &samples);
  }

  // convert raw input data such that categories of different
  // categorical features have unique indices
  void data_to_unique_categories(Tensor2<dtype> data, cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
