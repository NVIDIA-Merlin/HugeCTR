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
#include <embeddings/hybrid_embedding/model.hpp>
#include <embeddings/hybrid_embedding/utils.hpp>
#include <gpu_resource.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {
namespace hybrid_embedding {

// ===========================================================================================
// Frequent Compression
// ===========================================================================================

template <typename dtype>
struct FrequentEmbeddingCompressionView {
  const dtype* samples;
  bool* cache_masks;
  uint32_t *model_cache_indices, *model_cache_indices_offsets;
  uint32_t *network_cache_indices, *network_cache_indices_offsets;
  uint32_t *d_num_frequent_sample_indices, *frequent_sample_indices;
};

template <typename dtype>
class FrequentEmbeddingCompression {
  void calculate_frequent_sample_indices_temp_storage_bytes(const size_t local_samples_size);
  void calculate_model_cache_indices_temp_storage_bytes(const size_t num_frequent);
  void calculate_network_cache_indices_temp_storage_bytes(const size_t num_frequent);

  const Model<dtype>& model_;
  const Data<dtype>& data_;

  FrequentEmbeddingCompressionView<dtype>* device_indices_view_;

 public:
  // Role:
  //   push from the locally reduced gradient buffer => update embedding vector
  //   pull embedding vector from the model => update local cache
  //
  // Def:
  //  1 if frequent category is present in this network batch
  //  [size num_frequent]
  Tensor2<bool> cache_masks_;

  // model_cache_indices : list of cache indices of this frequent embedding model instance
  //                       for each mlp deep learning network.
  // Definition.
  //                       given the frequent embedding model of frequent embedding vectors
  //                       stored and updated by this instance, i.e. the range in
  //                       frequent_embedding_vectors
  //                         i * num_frequent /num_instances ... (i+1) * num_frequent /num_instances
  //                         - 1
  //                       for each network n, the range within model_cache_indices specified by
  //                         model_cache_indices_offsets_[n] .. model_cache_indices_offsets_[n] - 1
  //                       is the list of frequent cache indices that appear in network n.
  //
  // Role.
  //
  //       1. Forward-model :   cache indices into the frequent_embedding_vector array
  //                            for each send-message-buffer - per mlp network.
  //       2. Backward-model :  cache indices for each receive-message-buffer - mlp
  //
  Tensor2<uint32_t> model_cache_indices_;
  Tensor2<uint32_t> model_cache_indices_offsets_;

  // network_cache_indices : list of cache indices contained in this network for each
  //                         frequent embedding model instance
  // Def.
  //                         Given the mlp deep learning network samples for this instance,
  //                         - network n, sample_ids starting with i * batch_size / num_instances -
  //                         For each embedding model - model_id - list its cache indices that
  //                         are present within network n's samples. The range of these indices is
  //                         given by network_cache_indices_offsets_[i+1] ...
  //                         network_cache_indices_offsets_[i+1]
  // Role.
  //       1. Forward-network :   cache indices into the frequent_embedding_vector array
  //                              for each receive-message-buffer - per frequent embedding model
  //       2. Backward-network :  cache indices into the frequent_gradient_vectors_
  //                              for each send-message-buffer - mlp
  //
  Tensor2<uint32_t> network_cache_indices_;
  Tensor2<uint32_t> network_cache_indices_offsets_;

  // Role:
  //   from buffer => interaction layer
  //   sample gradients => gradient buffer
  //
  // Def:
  //   sample id's within this network batch
  //   containing frequent category [network batch size]
  // "Network side"
  Tensor2<uint32_t> d_num_frequent_sample_indices_;
  Tensor2<uint32_t> frequent_sample_indices_;

  // scratch buffers for index calculations
  Tensor2<char> frequent_sample_indices_temp_storage_;
  Tensor2<char> model_cache_indices_temp_storage_;
  Tensor2<char> network_cache_indices_temp_storage_;
  size_t frequent_sample_indices_temp_storage_bytes_;
  size_t model_cache_indices_temp_storage_bytes_;
  size_t network_cache_indices_temp_storage_bytes_;

  FrequentEmbeddingCompression(size_t max_num_frequent_categories, const Data<dtype>& data,
                               const Model<dtype>& model);

  void calculate_frequent_sample_indices(cudaStream_t stream);
  void calculate_model_cache_indices(size_t sm_count, cudaStream_t stream);
  void calculate_network_cache_mask(cudaStream_t stream);
  void calculate_network_cache_indices(cudaStream_t stream);
  void calculate_cache_masks(cudaStream_t stream);

  FrequentEmbeddingCompressionView<dtype>* get_device_view() { return device_indices_view_; };
  const Data<dtype>* get_data() { return &data_; }
};

// ===========================================================================================
// Infrequent Selection
// ===========================================================================================

template <typename dtype>
struct InfrequentEmbeddingSelectionView {
  const dtype* samples;
  uint32_t *model_indices, *model_indices_offsets;
  uint32_t *network_indices, *network_indices_offsets, *network_indices_src_model_id;
};

template <typename dtype>
class InfrequentEmbeddingSelection {
  void calculate_model_indices_temp_storage_bytes(size_t max_batch_size, size_t table_size);
  void calculate_network_indices_temp_storage_bytes(size_t max_batch_size, size_t table_size,
                                                    const uint32_t num_instances);

  const Model<dtype>& model_;
  const Data<dtype>& data_;
  InfrequentEmbeddingSelectionView<dtype>* device_indices_view_;

 public:
  // model_indices : list of samples indices of categories for which the embedding vectors are
  //                 stored in this infrequent embedding model instance.
  //                 Sample-id's for entire batch, i.e. sorted by mlp deep learning network.
  // Definition.
  //                 Given the infrequent embedding model of infrequent embedding vectors
  //                 stored and updated by this instance, sample indices for categories such
  //                 that
  //                     category_location[2*category] == model_id
  //                 for each network n, the range within model_cache_indices specified by
  //                    model_indices_offsets_[n] .. model_indices_offsets_[n+1] - 1
  //                 is the list of infrequent sample indices in network n.
  // Role.
  //       1. Forward-model :   indices in the samples array for each send-message-buffer
  //                            - per mlp network.
  //       2. Backward-model :  indices in the samples array for each receive-message-buffer
  //                            - per mlp network.
  Tensor2<uint32_t> model_indices_;
  Tensor2<uint32_t> model_indices_offsets_;
  // Tensor2<size_t> model_indices_sizes_;
  // Tensor2<size_t *> model_indices_sizes_ptrs_;

  // network_indices : list of sample indices of infrequent categories ordered per infrequent
  //                   embedding model - model_id - where they're stored.
  //                   Sample-id's for local batch (i.e sub-batch of this mlp network)
  // Definition.
  //                   Given the mlp deep learning network samples for this instance,
  //                   - network n, sample_ids starting with i * batch_size / num_instances -
  //                   For each embedding model - model_id - list its sample indices that
  //                   are present within network n's samples. The range of these indices is given
  //                   by
  //                     network_indices_offsets_[n] .. network_indices_offsets_[n+1] - 1
  // Role.
  //       1. Forward-network :   local sample indices for each receive-message-buffer
  //                              - per infrequent embedding model.
  //       2. Backward-network :  local sample indices for each send-message-buffer
  //                              - mlp
  Tensor2<uint32_t> network_indices_;
  Tensor2<uint32_t> network_indices_offsets_;
  Tensor2<uint32_t> network_indices_src_model_id_;
  // Tensor2<size_t> network_indices_sizes_;
  // Tensor2<size_t *> network_indices_sizes_ptrs_;

  // scratch buffers for index calculations
  /// TODO: if not overlapping, we can use the same storage
  Tensor2<char> model_indices_temp_storage_;
  size_t model_indices_temp_storage_bytes_;
  Tensor2<char> network_indices_temp_storage_;
  size_t network_indices_temp_storage_bytes_;

  InfrequentEmbeddingSelection(const Data<dtype>& data, const Model<dtype>& model);

  void calculate_model_indices(cudaStream_t stream);
  void calculate_network_indices(size_t sm_count, cudaStream_t stream);

  // For now these functions stay in InfreqeuentEmbedding
  //  since the communications can only use one offsets tensor
  // void calculate_model_indices_sizes_from_offsets(  size_t embedding_vec_bytes, cudaStream_t
  // stream); void calculate_network_indices_sizes_from_offsets(size_t embedding_vec_bytes,
  // cudaStream_t stream);

  InfrequentEmbeddingSelectionView<dtype>* get_device_view() { return device_indices_view_; }
  const Data<dtype>* get_data() { return &data_; }
};

// Single-stream version
template <typename dtype>
void compute_indices(FrequentEmbeddingCompression<dtype>& compression,
                     InfrequentEmbeddingSelection<dtype>& selection,
                     CommunicationType communication_type, bool compute_network_cache_indices,
                     cudaStream_t main_stream, int sm_count);

}  // namespace hybrid_embedding
}  // namespace HugeCTR
