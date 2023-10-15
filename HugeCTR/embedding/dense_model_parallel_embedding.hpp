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

struct DenseUniformModelParallelEmbeddingMeta {
  mutable int num_local_hotness_after_reduction_;
  mutable int num_local_hotness_before_reduction_;
  int global_hotness_;

  int num_lookup_;

  int ev_size_;

  int num_local_lookup_;
  int global_ev_offset_;
  std::vector<int> h_local_hotness_range_;

  std::vector<int> h_local_hotness_;

  std::vector<int> h_ev_start_indices_;

  DenseModelCommBufferAttr model_buffer_attr;

  DenseNetworkIndices network_indices;
  DenseNetworkBufferAttr network_buffer_attr;

  WgradAttr wgrad_attr;

  DenseUniformModelParallelEmbeddingMeta(std::shared_ptr<CoreResourceManager> core,
                                         const EmbeddingCollectionParam &ebc_param,
                                         size_t grouped_id);
};

class DenseUniformModelParallelEmbedding : public IGroupedEmbeddingOp {
  std::shared_ptr<CoreResourceManager> core_;
  DenseUniformModelParallelEmbeddingMeta meta_;

  DenseReductionIndices reduction_indices_;
  DenseMPLocalReduceIndexCalculation local_reduce_index_calculation_;
  LocalReduce local_reduce_;

  ModelForward model_forward_;
  NcclAll2AllComm all2all_comm_;
  NetworkForward network_forward_;

  NetworkBackward network_backward_;

  core23::Tensor embedding_vec_;  // storing lookup result (embedding vector addresses)

  DenseModelCommBuffer model_comm_buffer_;
  DenseNetworkBuffer network_buffer_;

  bool do_reduction_;
  void model_forward(const EmbeddingInput &embedding_input, ILookup *embedding_table,
                     int batch_size);

  void network_forward(const EmbeddingInput &embedding_input, EmbeddingOutput &embedding_output,
                       int batch_size);

  void backward_index_calculation(const EmbeddingInput &embedding_input, Wgrad &wgrad,
                                  int batch_size);

  void network_backward(const EmbeddingOutput &top_grad, const EmbeddingInput &embedding_input,
                        Wgrad &wgrad, int batch_size);

  void local_reduce(Wgrad &wgrad, int batch_size);

 public:
  DenseUniformModelParallelEmbedding(std::shared_ptr<CoreResourceManager> core,
                                     const EmbeddingCollectionParam &params, size_t grouped_id);

  void forward_per_gpu(Stage stage, const EmbeddingInput &embedding_input, ILookup *embedding_table,
                       EmbeddingOutput &embedding_output, int batch_size) override;

  void backward_per_gpu(Stage stage, const EmbeddingInput &embedding_input,
                        const EmbeddingOutput &top_grad, Wgrad &wgrad, int batch_size) override;

  const WgradAttr &get_wgrad_attr() const override { return meta_.wgrad_attr; };

  bool is_valid_stage(Stage stage) const override;
};

}  // namespace embedding
