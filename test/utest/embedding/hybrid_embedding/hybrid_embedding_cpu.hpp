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

#include <vector>

#include "HugeCTR/include/common.hpp"
#include "input_generator.hpp"

using namespace HugeCTR;
using namespace HugeCTR::hybrid_embedding;

namespace utils {
template <typename IntType>
constexpr static inline IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}
}  // namespace utils

template <typename dtype, typename emtype>
class HybridEmbeddingCpu {
 public:
  uint32_t num_instances;
  uint32_t num_nodes;
  uint32_t num_tables;
  float lr;

  uint32_t batch_size;
  uint32_t num_categories;
  uint32_t num_frequent;
  uint32_t embedding_vec_size;
  const std::vector<dtype>& category_location;
  const std::vector<dtype>& samples;

  uint32_t local_batch_size;
  uint32_t local_samples_size;

  std::vector<std::vector<uint32_t>> model_indices;
  std::vector<std::vector<uint32_t>> model_indices_offsets;
  std::vector<std::vector<uint32_t>> network_indices;
  std::vector<std::vector<uint32_t>> network_indices_offsets;
  std::vector<std::vector<uint32_t>> frequent_sample_indices;
  std::vector<std::vector<uint32_t>> model_cache_indices;
  std::vector<std::vector<uint32_t>> model_cache_indices_offsets;
  std::vector<std::vector<uint8_t>> network_cache_mask;
  std::vector<std::vector<uint32_t>> network_cache_indices;
  std::vector<std::vector<uint32_t>> network_cache_indices_offsets;

  std::vector<std::vector<float>> frequent_embedding_vectors;
  std::vector<std::vector<float>> infrequent_embedding_vectors;
  std::vector<std::vector<emtype>> gradients;
  std::vector<std::vector<emtype>> frequent_embedding_vectors_cache;

  std::vector<std::vector<emtype>> forward_sent_messages;
  std::vector<std::vector<emtype>> forward_received_messages;

  std::vector<std::vector<emtype>> backward_sent_messages;
  std::vector<std::vector<emtype>> backward_received_messages;

  std::vector<emtype> reduced_gradients;

  std::vector<std::vector<emtype>> interaction_layer_input;

  HybridEmbeddingCpu(const HybridEmbeddingConfig<dtype>& config, size_t batch_size,
                     const std::vector<dtype>& category_location,
                     const std::vector<dtype>& samples)
      : num_instances(config.num_instances),
        num_nodes(config.num_nodes),
        num_tables(config.num_tables),
        lr(config.lr),
        batch_size(batch_size),
        num_categories(config.num_categories),
        num_frequent(config.num_frequent),
        embedding_vec_size(config.embedding_vec_size),
        category_location(category_location),
        samples(samples),
        local_batch_size(utils::ceildiv<uint32_t>(batch_size, num_instances)),
        local_samples_size(local_batch_size * num_tables) {}

  void calculate_infrequent_model_indices();
  void calculate_infrequent_network_indices();
  void calculate_frequent_sample_indices();
  void calculate_frequent_model_cache_indices();
  void calculate_frequent_network_cache_indices();
  void calculate_frequent_network_cache_mask();

  void generate_embedding_vectors();
  void generate_gradients();

  void forward_a2a_messages();
  void forward_a2a_messages_hier();
  void backward_a2a_messages();
  void backward_a2a_messages_hier();

  void infrequent_update();
  void frequent_reduce_gradients();
  void frequent_update();
  void frequent_update_single_node();

  void forward_network();
  void frequent_forward_model();
};
