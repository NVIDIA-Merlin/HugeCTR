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
#include "operators/communication.hpp"
#include "operators/compress_offset.hpp"
#include "operators/model_backward.hpp"
#include "operators/model_forward.hpp"
#include "operators/mp_index_calculation.hpp"
#include "operators/network_backward.hpp"
#include "operators/network_forward.hpp"
namespace embedding {
using namespace core;

class UniformModelParallelEmbedding : public IGroupedEmbeddingOp {
  std::shared_ptr<CoreResourceManager> core_;
  UniformModelParallelEmbeddingMeta meta_;

  ModelIndexCalculation model_index_calculation_;
  ModelBackwardIndexCalculation model_backward_index_calculation_;
  CompressOffset compress_offset_;
  ModelForward model_forward_;
  NcclAll2AllComm all2all_comm_;
  NetworkForward network_forward_;

  NetworkBackward network_backward_;
  ModelBackward model_backward_;

  TensorList embedding_vec_;

  std::vector<Tensor> model_comm_buffer_list_;
  TensorList model_comm_buffer_;
  std::vector<Tensor> network_comm_buffer_list_;
  TensorList network_comm_buffer_;

  int batch_size_;
  Tensor bucket_range_, num_key_per_lookup_offset_, model_key_, model_offsets_;
  size_t num_model_key_;

  std::vector<size_t> get_model_comm_buffer_size(int universal_batch_size);
  void init_model_comm_buffer(int universal_batch_size, DataType emb_type);

  std::vector<size_t> get_network_comm_buffer_size(int universal_batch_size);
  void init_network_comm_buffer(int universal_batch_size, DataType emb_type);

 public:
  UniformModelParallelEmbedding(std::shared_ptr<CoreResourceManager> core,
                                const EmbeddingCollectionParam &params, size_t grouped_id);

  void forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                       ILookup *embedding_table, Tensor &output_buffer, int batch_size) override;

  void backward_per_gpu(const Tensor &top_grad, bool do_allreduce, Tensor *unique_key,
                        size_t *num_unique_key, Tensor *num_unique_key_per_table_offset,
                        size_t *num_table_offset, Tensor *table_id_list, Tensor *wgrad,
                        Tensor *wgrad_idx_offset) override;
};
}  // namespace embedding
