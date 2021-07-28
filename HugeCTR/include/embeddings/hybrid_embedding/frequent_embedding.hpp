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

// One FrequentEmbedding instance per gpu
template <typename dtype, typename emtype>
class FrequentEmbedding {
 public:
  // copy of the model parameters and the input data
  const Model<dtype> &model_;
  Data<dtype> data_;  // will be determined in the training procedure
  const Data<dtype> &data_train_;
  const Data<dtype> &data_evaluate_;
  const GPUResource &gpu_resource;

  // locally stored embedding vectors for the data-parallel part of the embedding for each table
  Tensor2<float> frequent_embedding_vectors_;
  Tensor2<emtype> frequent_embedding_vectors_cache_;

  Tensor2<emtype *> embedding_vectors_cache_pointers_;
  Tensor2<const emtype *> partial_gradients_pointers_;

  // locally stored reduced gradients into fp32 type
  Tensor2<float> float_frequent_gradients_;
  // buffer for communication can have fp16 type instead of fp32:
  // input for all-reduce
  Tensor2<emtype> frequent_gradients_;

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

  // scratch buffers for index calculations
  Tensor2<char> frequent_sample_indices_temp_storage_;
  Tensor2<char> model_cache_indices_temp_storage_;
  Tensor2<char> network_cache_indices_temp_storage_;
  size_t frequent_sample_indices_temp_storage_bytes;
  size_t model_cache_indices_temp_storage_bytes;
  size_t network_cache_indices_temp_storage_bytes;

  template <typename T>
  using BuffPtr = std::shared_ptr<BufferBlock2<T>>;
  BuffPtr<emtype> grouped_wgrad_buff_;

  uint32_t embedding_vec_size_;
  size_t max_num_frequent_categories_;
  void init();

  FrequentEmbedding(const Data<dtype> &data_train, const Data<dtype> &data_evaluate,
                    const Model<dtype> &model, const GPUResource &gpu_resource,
                    BuffPtr<emtype> &grouped_wgrad_buff, uint32_t embedding_vec_size,
                    size_t max_num_frequent_categories);
  ~FrequentEmbedding() {}

  void initialize_embedding_vectors(size_t grouped_wgrad_offset);
  void forward_model(cudaStream_t stream);
  void forward_model_eval(cudaStream_t stream);
  template <typename vectype>
  void forward_network_aux(const vectype *embedding_vectors, emtype *interaction_layer_input,
                           cudaStream_t stream);
  void forward_network(emtype *interaction_layer_input, bool from_cache, cudaStream_t stream);
  void update_model(float* dev_lr, float scale, cudaStream_t stream);
  void local_reduce(const emtype *gradients, cudaStream_t stream, bool reset_all = true);
  void update_model_direct(float* dev_lr, float scale, cudaStream_t stream);
  void calculate_frequent_sample_indices_temp_storage_bytes();
  void calculate_model_cache_indices_temp_storage_bytes();
  void calculate_network_cache_indices_temp_storage_bytes();
  void calculate_frequent_sample_indices(cudaStream_t stream);
  void calculate_model_cache_indices(cudaStream_t stream);
  void calculate_cache_masks(cudaStream_t stream);
  void calculate_network_cache_indices(cudaStream_t stream);

  template <typename T = emtype>
  typename std::enable_if<std::is_same<emtype, float>::value, Tensor2<T>>::type &get_gradients() {
    return float_frequent_gradients_;
  }

  template <typename T = emtype>
  typename std::enable_if<!std::is_same<T, float>::value, Tensor2<T>>::type &get_gradients() {
    return frequent_gradients_;
  }

  template <typename T = emtype>
  typename std::enable_if<std::is_same<T, float>::value, Tensor2<T>>::type
  get_embedding_vectors_cache() {
    return frequent_embedding_vectors_;
  }

  template <typename T = emtype>
  typename std::enable_if<!std::is_same<T, float>::value, Tensor2<T>>::type
  get_embedding_vectors_cache() {
    return frequent_embedding_vectors_cache_;
  }

  class ExternalManagedBuffer : public HugeCTR::TensorBuffer2 {
   public:
    ExternalManagedBuffer(void *ptr) : ptr_(ptr) {}
    bool allocated() const override { return true; }
    void *get_ptr() override { return ptr_; }

   private:
    void *ptr_;
  };
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
