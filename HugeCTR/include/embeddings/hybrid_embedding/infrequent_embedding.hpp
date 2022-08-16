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
template <typename dtype>
class InfrequentEmbeddingBase {
 protected:
  const Data<dtype> *data_ = nullptr;
  InfrequentEmbeddingSelectionView<dtype> *indices_view_ = nullptr;

 public:
  // Infrequent indices and device pointer!
  InfrequentEmbeddingSelection<dtype> *indices_;

  void set_current_indices(InfrequentEmbeddingSelection<dtype> *indices, cudaStream_t stream);
  InfrequentEmbeddingBase();
  virtual ~InfrequentEmbeddingBase();

  InfrequentEmbeddingBase(const InfrequentEmbeddingBase &other);

  InfrequentEmbeddingBase &operator=(const InfrequentEmbeddingBase &other) {
    if (this == &other) {
      return *this;
    }

    HCTR_LIB_THROW(cudaMalloc(&indices_view_, sizeof(*indices_view_)));

    HCTR_LIB_THROW(cudaMemcpy(indices_view_, other.indices_view_, sizeof(*indices_view_),
                              cudaMemcpyDeviceToDevice));

    return *this;
  }
};

template <typename dtype, typename emtype>
class InfrequentEmbedding_NVLink_SingleNode : public InfrequentEmbeddingBase<dtype> {
 public:
  using InfrequentEmbeddingBase<dtype>::data_;

  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  const Model<dtype> &model_;
  const GPUResource &gpu_resource_;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for
  // each table
  Tensor2<float> infrequent_embedding_vectors_;

  Tensor2<emtype *> interaction_layer_input_pointers_train_;
  Tensor2<emtype *> interaction_layer_input_pointers_eval_;
  Tensor2<const emtype *> gradients_pointers_;

  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_;

  void init_pointers(int local_gpu_count, const cudaStream_t stream,
                     std::vector<emtype *> &interaction_layer_input_pointers_train,
                     std::vector<emtype *> &interaction_layer_input_pointers_eval,
                     std::vector<const emtype *> &gradients_pointers);

  InfrequentEmbedding_NVLink_SingleNode(Model<dtype> &model, GPUResource &gpu_resource,
                                        size_t embedding_vec_size);

  ~InfrequentEmbedding_NVLink_SingleNode() {}

  void initialize_embedding_vectors(const std::vector<size_t> &table_sizes);
  void forward_network_direct(bool is_train, cudaStream_t stream);
  void update_model_direct(float *dev_lr, float scale, cudaStream_t stream);
};

template <typename dtype, typename emtype>
class InfrequentEmbedding_IB_NVLINK : public InfrequentEmbeddingBase<dtype> {
 public:
  using InfrequentEmbeddingBase<dtype>::data_;

  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  const Model<dtype> &model_;
  const GPUResource &gpu_resource_;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for
  // each table
  Tensor2<float> infrequent_embedding_vectors_;

  // Tensors to be passed to the hierarchical comms
  // TODO: move these to the index containers
  Tensor2<uint32_t> network_indices_offsets_, model_indices_offsets_;

  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_;

  // private:
  std::unique_ptr<AllToAllStorage<emtype>> infrequent_forward_comm_buffers_,
      infrequent_backward_comm_buffers_;
  std::unique_ptr<Communication> infrequent_forward_comms_, infrequent_backward_comms_;

  // requires model_ and data_ to be set
  InfrequentEmbedding_IB_NVLINK(Model<dtype> &model, GPUResource &gpu_resource,
                                size_t embedding_vec_size);

  //~InfrequentEmbedding_IB_NVLINK(){};

  void init_comms(size_t embedding_vec_size, const GPUResource *gpu_resource,
                  GeneralBuffer2<CudaAllocator> *i_buf, size_t max_buf_size);
  void initialize_embedding_vectors(const std::vector<size_t> &table_sizes);
  void forward_model(emtype *message_buffer, cudaStream_t stream);
  void forward_network(const emtype *message_buffer, emtype *interaction_layer_input,
                       cudaStream_t stream);
  void forward(emtype *output_ptr, cudaStream_t stream);
  void update_network(const emtype *gradients, emtype *message_buffer, cudaStream_t stream);
  void update_model(const emtype *message_buffer, float *dev_lr, float scale, cudaStream_t stream);

  const uint32_t *get_model_indices_offsets_ptr() { return model_indices_offsets_.get_ptr(); }
  const uint32_t *get_network_indices_offsets_ptr() { return network_indices_offsets_.get_ptr(); }
};

template <typename dtype, typename emtype>
class InfrequentEmbedding_IB_NVLink_Hier : public InfrequentEmbeddingBase<dtype> {
 public:
  using InfrequentEmbeddingBase<dtype>::data_;

  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  const Model<dtype> &model_;
  const GPUResource &gpu_resource_;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for
  // each table
  Tensor2<float> infrequent_embedding_vectors_;

  // Communication buffer sizes
  dtype max_num_infrequent_per_batch_;
  dtype max_num_infrequent_per_train_batch_;

  // Tensors to be passed to the hierarchical comms
  // TODO: move these to the index containers
  Tensor2<size_t> network_indices_sizes_, model_indices_sizes_;
  Tensor2<size_t *> network_indices_sizes_ptrs_, model_indices_sizes_ptrs_;

  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_;

  std::unique_ptr<AllToAllStorage<emtype>> infrequent_forward_comm_buffers_,
      infrequent_backward_comm_buffers_;
  std::unique_ptr<Communication> infrequent_forward_comms_, infrequent_backward_comms_;

  // requires model_ and data_ to be set
  InfrequentEmbedding_IB_NVLink_Hier(Model<dtype> &model, GPUResource &gpu_resource,
                                     size_t embedding_vec_size);
  //~InfrequentEmbedding_IB_NVLink_Hier(){};

  void init_comms(int64_t max_num_infrequent_samples, size_t slot_num, size_t embedding_vec_size,
                  GeneralBuffer2<CudaAllocator> *buf_ptr, size_t batch_size_true,
                  size_t batch_size_false, size_t local_gpu_count);
  void forward_model(cudaStream_t stream);
  void infrequent_wait_completion(cudaStream_t stream);
  void infrequent_hier_forward_network(emtype *output_ptr, cudaStream_t stream);
  void initialize_embedding_vectors(const std::vector<size_t> &table_sizes);
  void fused_intra_forward_model(emtype **message_buffer, cudaStream_t stream);
  void hier_forward_network(const emtype *message_buffer, emtype *interaction_layer_input,
                            cudaStream_t stream);
  void fused_intra_update_network(const emtype *gradients, emtype **message_buffer,
                                  cudaStream_t stream);
  void hier_update_model(const emtype *message_buffer, float *dev_lr, float scale,
                         cudaStream_t stream);
  void calculate_model_indices_sizes_from_offsets(cudaStream_t stream);
  void calculate_network_indices_sizes_from_offsets(cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR
