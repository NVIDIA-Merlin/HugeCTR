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

#include <random>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
struct HybridEmbeddingConfig {
  size_t num_nodes;
  size_t num_instances;
  size_t num_tables;
  size_t embedding_vec_size;
  dtype num_categories;
  dtype num_frequent;
  float lr;
  CommunicationType comm_type;
};

template <typename dtype>
class HybridEmbeddingInputGenerator {
 public:
  HybridEmbeddingInputGenerator(size_t seed) : gen_(seed) {}
  HybridEmbeddingInputGenerator(HybridEmbeddingConfig<dtype> config, size_t seed);
  HybridEmbeddingInputGenerator(HybridEmbeddingConfig<dtype> config,
                                const std::vector<size_t> &table_sizes, size_t seed);
  // Multiple calls return different data

  // By default the data is provided in the 'raw' format: each data point is
  // a category which is indexed according to the table it belongs to.
  // Each sample contains <number of tables> elements and its
  // value lies within the integer range [0, number of categories in category feature)

  /// @param batch_size number of samples to return
  /// @param num_categories required sum of table sizes
  /// @param num_tables required number of tables
  /// @param flatten_input indicator whether generated categories have an associated unique value
  std::vector<dtype> generate_categorical_input(size_t batch_size, size_t num_tables);
  // _flattened means that the category indices are unique
  // (i.e., table offsets are added to the raw data)
  std::vector<dtype> generate_flattened_categorical_input(size_t batch_size, size_t num_tables);

  // regenerate data with precalculated table_sizes_
  std::vector<dtype> generate_categorical_input(size_t batch_size);
  std::vector<dtype> generate_flattened_categorical_input(size_t batch_size);

  void generate_categorical_input(dtype *batch, size_t batch_size);
  void generate_flattened_categorical_input(dtype *batch, size_t batch_size);
  void generate_category_location();

  // Multiple calls return the same data
  std::vector<dtype> &get_category_location();
  std::vector<size_t> &get_table_sizes();

 private:
  HybridEmbeddingConfig<dtype> config_;
  std::vector<std::vector<double>> embedding_prob_distribution_;
  std::vector<size_t> table_sizes_;
  size_t seed_;
  std::mt19937 gen_;

  std::vector<dtype> category_location_;
  std::vector<std::vector<size_t>> embedding_shuffle_args;

  void generate_uniform_rand_table_sizes(size_t num_categories = 0, size_t num_tables = 0);
  static std::vector<size_t> generate_rand_table_sizes(size_t num_tables,
                                                       size_t embedding_vec_size = 128,
                                                       double max_mem = 8.e9);
  void create_probability_distribution();
  void generate_categories(dtype *data, size_t batch_size, bool normalized);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR