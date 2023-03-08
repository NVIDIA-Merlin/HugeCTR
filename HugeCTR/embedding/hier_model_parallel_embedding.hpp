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

#include "HugeCTR/embedding/gpu_barrier/gpu_barrier.hpp"
#include "common.hpp"
#include "embedding.hpp"
#include "model_parallel_embedding.hpp"
#include "operators/hier_model_backward.hpp"
#include "operators/hier_model_forward.hpp"

namespace embedding {
using namespace core;

struct HierModelParallelEmbeddingMeta {
  mutable std::vector<int> h_local_hotness_list_;
  mutable int num_local_hotness_;

  int num_local_lookup_;

  std::vector<int> h_local_table_id_list_;
  core23::Tensor d_local_table_id_list_;

  ModelCommBufferAttr model_buffer_attr;

  std::vector<std::vector<int>> h_lookup_ids_in_current_rail;
  NetworkIndices hier_network_indices;
  NetworkBufferAttr hier_network_buffer_attr;

  WgradAttr wgrad_attr;

  std::vector<int> h_table_id_to_global_start_indices;
  core23::Tensor table_id_to_global_start_indices;

  EmbeddingOutputAttr output_attr;
  IntraModelCommBufferAttr intra_model_buffer_attr;
  std::vector<IntraModelReductionBufferAttr> intra_model_reduction_buffer_attr_in_all_nodes;

  HierModelParallelEmbeddingMeta(std::shared_ptr<CoreResourceManager> core,
                                 const EmbeddingCollectionParam &params, size_t grouped_id);

  void update_mutable_meta(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param, size_t grouped_id) const;
};

class HierModelParallelEmbedding : public IGroupedEmbeddingOp {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  HierModelParallelEmbeddingMeta meta_;

  ReductionIndices reduction_indices_;
  MPLocalReduceIndexCalculation local_reduce_index_calculation_;
  LocalReduce local_reduce_;

  CompressOffset compress_offset_;
  IntraModelForward intra_model_forward_;
  NcclAll2AllComm all2all_comm_;
  NetworkForward network_forward_;

  NetworkBackward network_backward_;
  IntraModelBackward intra_model_backward_;

  core23::Tensor embedding_vec_;

  ModelCommBuffer model_comm_buffer_;
  NetworkBuffer network_buffer_;
  IntraModelCommBuffer intra_model_comm_buffer_;
  IntraModelReductionBuffer intra_reduction_buffer_;

  HugeCTR::GPUBarrier *gpu_barrier_;

 public:
  HierModelParallelEmbedding(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam &params, size_t grouped_id);

  void forward_per_gpu(const EmbeddingInput &embedding_input, ILookup *embedding_table,
                       EmbeddingOutput &embedding_output, int batch_size) override;

  void backward_per_gpu(const EmbeddingInput &embedding_input, const EmbeddingOutput &top_grad,
                        Wgrad &wgrad, int batch_size) override;

  const WgradAttr &get_wgrad_attr() const override { return meta_.wgrad_attr; }

  IntraModelCommBuffer *get_intra_model_comm_buffer() { return &intra_model_comm_buffer_; }

  ModelCommBuffer *get_model_comm_buffer() { return &model_comm_buffer_; }

  void set_gpu_barrier(HugeCTR::GPUBarrier *gpu_barrier) { gpu_barrier_ = gpu_barrier; }
};

}  // namespace embedding
