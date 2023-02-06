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

#include <embedding/common.hpp>
#include <embedding/embedding.hpp>
#include <embedding/operators/communication.hpp>
#include <embedding/operators/compress_offset.hpp>
#include <embedding/operators/index_calculation.hpp>
#include <embedding/operators/model_backward.hpp>
#include <embedding/operators/model_forward.hpp>
#include <embedding/operators/mp_index_calculation.hpp>
#include <embedding/operators/network_backward.hpp>
#include <embedding/operators/network_forward.hpp>

namespace embedding {
using namespace core;

struct UniformModelParallelEmbeddingMeta {
  mutable std::vector<int> h_hotness_list_;
  mutable int hotness_sum_;
  mutable std::vector<int> h_local_hotness_list_;
  mutable int num_local_hotness_;

  int num_lookup_;

  std::vector<int> h_ev_size_list_;
  std::vector<int> h_ev_size_offset_;
  Tensor d_ev_size_offset_;
  int max_ev_size_;
  int num_sms_;
  KernelParams kernel_params;

  std::vector<char> h_combiner_list_;
  Tensor d_combiner_list_;

  int num_local_lookup_;
  std::vector<int> h_local_shard_id_list_;
  Tensor d_local_shard_id_list_;

  std::vector<int> h_local_num_shards_list_;
  Tensor d_local_num_shards_list_;

  std::vector<int> h_local_table_id_list_;
  Tensor d_local_table_id_list_;

  std::vector<int> h_local_lookup_id_list_;
  Tensor d_local_lookup_id_list_;

  std::vector<int> h_local_ev_size_list_;
  Tensor d_local_ev_size_list_;

  std::vector<int> h_local_ev_size_offset_;
  Tensor d_local_ev_size_offset_;

  ModelCommBufferAttr model_buffer_attr;

  std::vector<std::vector<int>> h_global_lookup_id_list_;

  NetworkIndices network_indices;
  NetworkBufferAttr network_buffer_attr;

  WgradAttr wgrad_attr;

  std::vector<int> h_table_id_to_global_start_indices;
  Tensor table_id_to_global_start_indices;

  UniformModelParallelEmbeddingMeta(std::shared_ptr<CoreResourceManager> core,
                                    const EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void update_mutable_meta(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param, size_t grouped_id) const;
};

class UniformModelParallelEmbedding : public IGroupedEmbeddingOp {
  std::shared_ptr<CoreResourceManager> core_;
  UniformModelParallelEmbeddingMeta meta_;

  ReductionIndices reduction_indices_;
  MPLocalReduceIndexCalculation local_reduce_index_calculation_;
  LocalReduce local_reduce_;

  CompressOffset compress_offset_;
  ModelForward model_forward_;
  NcclAll2AllComm all2all_comm_;
  NetworkForward network_forward_;

  NetworkBackward network_backward_;

  TensorList embedding_vec_;

  ModelCommBuffer model_comm_buffer_;
  NetworkBuffer network_buffer_;

  std::vector<size_t> get_model_comm_buffer_size(int universal_batch_size);

  std::vector<size_t> get_network_comm_buffer_size(int universal_batch_size);

 public:
  UniformModelParallelEmbedding(std::shared_ptr<CoreResourceManager> core,
                                const EmbeddingCollectionParam &params, size_t grouped_id);

  void forward_per_gpu(const EmbeddingInput &embedding_input, ILookup *embedding_table,
                       EmbeddingOutput &embedding_output, int batch_size) override;

  void backward_per_gpu(const EmbeddingInput &embedding_input, const EmbeddingOutput &top_grad,
                        Wgrad &wgrad, int batch_size) override;

  const WgradAttr &get_wgrad_attr() const override { return meta_.wgrad_attr; }
};

}  // namespace embedding
