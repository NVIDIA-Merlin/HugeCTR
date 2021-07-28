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

#include <memory>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype, typename emtype>
class InfrequentEmbedding {
 public:
  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  const Model<dtype> &model_;
  Data<dtype> data_;  // will be determined in the training procedure
  const Data<dtype> &data_train_;
  const Data<dtype> &data_evaluate_;
  const GPUResource &gpu_resource;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for
  // each table
  Tensor2<float> infrequent_embedding_vectors_;

  Tensor2<emtype *> interaction_layer_input_pointers_train_;
  Tensor2<emtype *> interaction_layer_input_pointers_eval_;
  Tensor2<const emtype *> gradients_pointers_;

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
  Tensor2<size_t> model_indices_sizes_;
  Tensor2<size_t *> model_indices_sizes_ptrs_;

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
  Tensor2<size_t> network_indices_sizes_;
  Tensor2<size_t *> network_indices_sizes_ptrs_;

  // scratch buffers for index calculations
  /// TODO: if not overlapping, we can use the same storage
  Tensor2<char> model_indices_temp_storage_;
  size_t model_indices_temp_storage_bytes;
  Tensor2<char> network_indices_temp_storage_;
  size_t network_indices_temp_storage_bytes;

  // Communication buffer sizes
  dtype max_num_infrequent_per_batch_;
  dtype max_num_infrequent_per_train_batch_;

  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_;
  // requires model_ and data_ to be set
  void init();
  InfrequentEmbedding(const Data<dtype> &data_train, const Data<dtype> &data_evaluate,
                      const Model<dtype> &model, const GPUResource &gpu_resource,
                      uint32_t embedding_vec_size);
  ~InfrequentEmbedding(){};

  void initialize_embedding_vectors();
  void forward_model(emtype *message_buffer, cudaStream_t stream);
  void fused_intra_forward_model(emtype **message_buffer, cudaStream_t stream);
  void forward_network(const emtype *message_buffer, emtype *interaction_layer_input,
                       cudaStream_t stream);
  void hier_forward_network(const emtype *message_buffer, emtype *interaction_layer_input,
                            cudaStream_t stream);
  void forward_network_direct(bool is_train, cudaStream_t stream);
  void update_network(const emtype *gradients, emtype *message_buffer, cudaStream_t stream);
  void fused_intra_update_network(const emtype *gradients, emtype **message_buffer,
                                  cudaStream_t stream);
  void update_model(const emtype *message_buffer, float* dev_lr, float scale, cudaStream_t stream);
  void hier_update_model(const emtype *message_buffer, float* dev_lr, float scale,
                         cudaStream_t stream);
  void update_model_direct(float* dev_lr, float scale, cudaStream_t stream);
  void calculate_model_indices_temp_storage_bytes();
  void calculate_network_indices_temp_storage_bytes();
  void calculate_model_indices(cudaStream_t stream);
  void calculate_network_indices(cudaStream_t stream);
  void calculate_model_indices_sizes_from_offsets(cudaStream_t stream);
  void calculate_network_indices_sizes_from_offsets(cudaStream_t stream);

  const uint32_t *get_model_indices_offsets_ptr() { return model_indices_offsets_.get_ptr(); }
  const uint32_t *get_network_indices_offsets_ptr() { return network_indices_offsets_.get_ptr(); }
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
