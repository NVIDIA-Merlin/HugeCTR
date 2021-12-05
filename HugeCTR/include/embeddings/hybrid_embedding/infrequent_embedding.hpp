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
#include "hybrid_indices.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

// In order to use it easier in the IndicesContainer
template<typename dtype> 
class InfrequentEmbeddingBase {
protected:
  const Data<dtype>* data_ = nullptr;
  InfrequentEmbeddingSelectionView<dtype> *indices_view_ = nullptr;

public:
  // Infrequent indices and device pointer!
  InfrequentEmbeddingSelection<dtype> *indices_;

  void set_current_indices(InfrequentEmbeddingSelection<dtype> *indices, cudaStream_t stream);
  InfrequentEmbeddingBase();
  ~InfrequentEmbeddingBase();
};

template <typename dtype, typename emtype>
class InfrequentEmbedding : public InfrequentEmbeddingBase<dtype> {  
 public:
   using InfrequentEmbeddingBase<dtype>::data_;

  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  const Model<dtype> &model_;
  const GPUResource &gpu_resource;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for
  // each table
  Tensor2<float> infrequent_embedding_vectors_;

  Tensor2<emtype *> interaction_layer_input_pointers_train_;
  Tensor2<emtype *> interaction_layer_input_pointers_eval_;
  Tensor2<const emtype *> gradients_pointers_;

  // Communication buffer sizes
  dtype max_num_infrequent_per_batch_;
  dtype max_num_infrequent_per_train_batch_;

  // Tensors to be passed to the hierarchical comms
  // TODO: move these to the index containers
  Tensor2<uint32_t> network_indices_offsets_, model_indices_offsets_; 
  Tensor2<size_t> network_indices_sizes_, model_indices_sizes_; 
  Tensor2<size_t *> network_indices_sizes_ptrs_, model_indices_sizes_ptrs_;

  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_;
  // requires model_ and data_ to be set
  void init();
  InfrequentEmbedding(const Model<dtype> &model, const GPUResource &gpu_resource,
                      uint32_t embedding_vec_size);
  ~InfrequentEmbedding(){};

  void initialize_embedding_vectors(const std::vector<size_t>& table_sizes);
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
  void update_model(const emtype *message_buffer, float *dev_lr, float scale, cudaStream_t stream);
  void hier_update_model(const emtype *message_buffer, float *dev_lr, float scale,
                         cudaStream_t stream);
  void update_model_direct(float *dev_lr, float scale, cudaStream_t stream);

  void calculate_model_indices_sizes_from_offsets(cudaStream_t stream);
  void calculate_network_indices_sizes_from_offsets(cudaStream_t stream);

  const uint32_t *get_model_indices_offsets_ptr() { return model_indices_offsets_.get_ptr(); }
  const uint32_t *get_network_indices_offsets_ptr() { return network_indices_offsets_.get_ptr(); }
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
