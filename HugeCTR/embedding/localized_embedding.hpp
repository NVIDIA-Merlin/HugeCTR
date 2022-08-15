/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "common.hpp"
#include "embedding.hpp"
#include "embedding_data.hpp"
#include "operators/communication.hpp"
#include "operators/compress_offset.hpp"
#include "operators/model_backward.hpp"
#include "operators/model_forward.hpp"
#include "operators/mp_index_calculation.hpp"
#include "operators/network_backward.hpp"
#include "operators/network_forward.hpp"
namespace embedding {
using namespace core;

class UniformLocalizedEmbeddingForward : public IEmbeddingForward {
  std::shared_ptr<CoreResourceManager> core_;
  const GlobalEmbeddingData &global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;

  ModelIndexCalculation model_index_calculation_;
  ModelBackwardIndexCalculation model_backward_index_calculation_;
  CompressOffset compress_offset_;
  ModelForward model_forward_;
  NcclAll2AllComm all2all_comm_;
  NetworkForward network_forward_;
  AverageCominber average_combiner_;

  RaggedNetworkIndex ragged_network_index_;
  RaggedNetworkBuffer ragged_network_buffer_;

  TensorList embedding_vec_;

  std::vector<Tensor> model_comm_buffer_list_;
  TensorList model_comm_buffer_;

  std::vector<size_t> get_model_comm_buffer_size(int universal_batch_size);
  void init_model_comm_buffer(int universal_batch_size, DataType emb_type);

  std::vector<size_t> get_network_comm_buffer_size(int universal_batch_size);
  void init_network_comm_buffer(int universal_batch_size, DataType emb_type);

 public:
  UniformLocalizedEmbeddingForward(std::shared_ptr<CoreResourceManager> core,
                                   const EmbeddingCollectionParam &params,
                                   const GlobalEmbeddingData &global_embedding_data,
                                   const EmbeddingShardingParam &embedding_sharding_param);

  void forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                       const Tensor &sparse_weight, ILookup *embedding_table, Tensor &output_buffer,
                       ContextContainer *context_container) override;
};

class UniformLocalizedEmbeddingBackward : public IEmbeddingBackward {
  std::shared_ptr<CoreResourceManager> core_;
  const GlobalEmbeddingData &global_embedding_data_;
  LocalEmbeddingData local_embedding_data_;

  NetworkBackward network_backward_;
  NcclAll2AllComm all2all_comm_;
  ModelBackward model_backward_;
  AverageCominber average_combiner_;

 public:
  UniformLocalizedEmbeddingBackward(std::shared_ptr<CoreResourceManager> core,
                                    const EmbeddingCollectionParam &params,
                                    const GlobalEmbeddingData &global_embedding_data,
                                    const EmbeddingShardingParam &embedding_sharding_param);

  void backward_per_gpu(ContextContainer *context_container, const Tensor &top_grad,
                        bool do_allreduce, Tensor *unique_key, size_t *num_unique_key,
                        Tensor *unique_id_space_offset, size_t *num_unique_key_id_space_offset,
                        Tensor *grad_ev, Tensor *unique_dst_idx) override;
};
}  // namespace embedding
