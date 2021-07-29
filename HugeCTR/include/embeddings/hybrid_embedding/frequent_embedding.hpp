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
#include "hybrid_indices.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

// TODO sort out public/private fields
// In order to use it easier in the IndicesContainer
template<typename dtype> 
class FrequentEmbeddingBase {
protected:
  const Data<dtype>* data_ = nullptr;
  FrequentEmbeddingCompressionView<dtype> *indices_view_ = nullptr;

public:
  // Frequent indices and device pointer!
  FrequentEmbeddingCompression<dtype> *indices_;

  void set_current_indices(FrequentEmbeddingCompression<dtype> *indices, cudaStream_t stream);
  FrequentEmbeddingBase();
  ~FrequentEmbeddingBase();
};

// One FrequentEmbedding instance per gpu
template <typename dtype, typename emtype>
class FrequentEmbedding : public FrequentEmbeddingBase<dtype> {
 public:
  using FrequentEmbeddingBase<dtype>::data_;

  // copy of the model parameters and the input data
  const Model<dtype> &model_;
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

  template <typename T>
  using BuffPtr = std::shared_ptr<BufferBlock2<T>>;
  BuffPtr<emtype> grouped_wgrad_buff_;

  uint32_t embedding_vec_size_;
  size_t max_num_frequent_categories_;
  void init();

  FrequentEmbedding(const Model<dtype> &model, const GPUResource &gpu_resource,
                    BuffPtr<emtype> &grouped_wgrad_buff, uint32_t embedding_vec_size,
                    size_t max_num_frequent_categories);
  ~FrequentEmbedding() {}

  void initialize_embedding_vectors(const std::vector<size_t>& table_sizes, size_t grouped_wgrad_offset);
  void forward_model(cudaStream_t stream);
  void forward_model_eval(cudaStream_t stream);
  template <typename vectype>
  void forward_network_aux(const vectype *embedding_vectors, emtype *interaction_layer_input,
                           cudaStream_t stream);
  void forward_network(emtype *interaction_layer_input, bool from_cache, cudaStream_t stream);
  void update_model(float *dev_lr, float scale, cudaStream_t stream);
  void local_reduce(const emtype *gradients, cudaStream_t stream, bool reset_all = true);
  void update_model_direct(float *dev_lr, float scale, cudaStream_t stream);

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
