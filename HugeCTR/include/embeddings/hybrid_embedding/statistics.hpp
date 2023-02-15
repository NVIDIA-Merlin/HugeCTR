/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <common.hpp>
#include <embeddings/hybrid_embedding/data.hpp>
#include <general_buffer2.hpp>
#include <numeric>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

namespace hybrid_embedding {

// depends on : data object
// => allocate(Data data)

template <typename dtype>
struct Statistics {
 public:
  Statistics()
      : num_samples(0),
        num_tables(0),
        num_instances(0),
        num_categories(0),
        num_unique_categories(0) {}
  ~Statistics() {}
  Statistics(dtype num_samples_in, size_t num_tables_in, size_t num_instances_in,
             dtype num_categories_in)
      : num_samples(num_samples_in),
        num_tables(num_tables_in),
        num_instances(num_instances_in),
        num_categories(num_categories_in),
        num_unique_categories(0) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    reserve(buf);
    buf->allocate();
  }
  Statistics(dtype num_samples_in, size_t num_tables_in, size_t num_instances_in,
             dtype num_categories_in, std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf)
      : num_samples(num_samples_in),
        num_tables(num_tables_in),
        num_instances(num_instances_in),
        num_unique_categories(0) {
    reserve(buf);
  }
  Statistics(const Data<dtype> &data, size_t num_instances_in)
      : num_samples(data.batch_size * data.num_iterations * data.table_sizes.size()),
        num_tables(data.table_sizes.size()),
        num_instances(num_instances_in),
        num_categories(std::accumulate(data.table_sizes.begin(), data.table_sizes.end(), (dtype)0)),
        num_unique_categories(0) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    reserve(buf);
    buf->allocate();
  }
  Statistics(const Data<dtype> &data, size_t num_instances_in,
             std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf)
      : num_samples(data.batch_size * data.num_iterations * data.table_sizes.size()),
        num_tables(data.table_sizes.size()),
        num_instances(num_instances_in),
        num_categories(std::accumulate(data.table_sizes.begin(), data.table_sizes.end(), 0)),
        num_unique_categories(0) {
    reserve(buf);
  }

  void reserve(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
    buf->reserve({num_samples, 1}, &categories_sorted);
    buf->reserve({num_samples, 1}, &counts_sorted);
    buf->reserve({num_tables + 1, 1}, &table_offsets);
    buf->reserve({num_tables + 1, 1}, &infrequent_model_table_offsets);
    buf->reserve({num_instances * (num_tables + 1), 1}, &frequent_model_table_offsets);
    reserve_temp_storage(buf);
  }

  size_t num_samples;  // input
  size_t num_tables;
  size_t num_instances;
  dtype num_categories;
  uint32_t num_unique_categories;  // to be calculated

  // top categories sorted by count
  Tensor2<dtype> categories_sorted;
  Tensor2<uint32_t> counts_sorted;
  Tensor2<dtype> table_offsets;  // cumulative sum of table_sizes
  Tensor2<dtype> infrequent_model_table_offsets;
  Tensor2<dtype> frequent_model_table_offsets;
  std::vector<Tensor2<unsigned char>> sort_categories_by_count_temp_storages_;
  std::vector<Tensor2<unsigned char>> calculate_frequent_categories_temp_storages_;
  std::vector<Tensor2<unsigned char>> calculate_infrequent_categories_temp_storages_;
  void reserve_temp_storage(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf);
  void sort_categories_by_count(const dtype *samples, size_t num_samples, dtype *categories_sorted,
                                uint32_t *counts_sorted, uint32_t &num_unique_categories,
                                cudaStream_t stream);
  void sort_categories_by_count(const Tensor2<dtype> &samples, cudaStream_t stream);
  void calculate_frequent_and_infrequent_categories(
      dtype *frequent_categories, dtype *infrequent_categories, dtype *category_location,
      const size_t num_frequent, const size_t num_infrequent, cudaStream_t stream);
  void calculate_infrequent_model_table_offsets(
      std::vector<dtype> &h_infrequent_model_table_offsets, const dtype *infrequent_categories,
      const Tensor2<dtype> &category_location, uint32_t global_instance_id,
      const dtype num_infrequent, cudaStream_t stream);
  void calculate_frequent_model_table_offsets(std::vector<dtype> &h_frequent_model_table_offsets,
                                              const dtype *frequent_categories,
                                              const dtype num_frequent, cudaStream_t stream);
  void revoke_temp_storage() {
    sort_categories_by_count_temp_storages_.clear();
    calculate_frequent_categories_temp_storages_.clear();
    calculate_infrequent_categories_temp_storages_.clear();
  }
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
