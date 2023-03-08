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
#include <embedding/operators/model_forward.hpp>

#include "HugeCTR/core/core.hpp"
#include "HugeCTR/embedding/common.hpp"
#include "network_forward.hpp"

namespace embedding {

struct IntraModelCommBufferAttr {
  std::vector<std::vector<int>> h_lookup_ids_in_current_node;

  std::vector<std::vector<int>> h_id_to_ev_size_in_current_node;
  std::vector<core23::Tensor> id_to_ev_size_in_current_node_list;
  core23::Tensor id_to_ev_size_in_current_node;

  std::vector<std::vector<int>> h_id_to_ev_start_indices_in_current_node;
  std::vector<core23::Tensor> id_to_ev_start_indices_in_current_node_list;
  core23::Tensor id_to_ev_start_indices_in_current_node;

  core23::DataType type;

  int num_local_lookup;
  core23::Tensor id_to_ev_size_in_current_gpu;
  core23::Tensor id_to_ev_start_indices_in_current_gpu;
  int max_ev_size;

  IntraModelCommBufferAttr(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param, size_t grouped_id);
};

struct IntraModelCommBuffer {
  std::vector<core23::Tensor> local_datas;  // num of local gpus
  core23::Tensor local_datas_device_view;
  core23::Tensor peer_data;

  const IntraModelCommBufferAttr &attr;

  IntraModelCommBuffer(std::shared_ptr<CoreResourceManager> core,
                       const IntraModelCommBufferAttr &attr, int batch_size);
};

void collective_init_peer_buffer(const std::vector<std::shared_ptr<CoreResourceManager>> &cores,
                                 std::vector<IntraModelCommBuffer *> &intra_model_comm_buffers);

struct IntraModelReductionBufferAttr {
  NetworkIndices indices;

  std::vector<int> h_inter_id_to_ev_size;
  core23::Tensor id_to_ev_size;

  std::vector<int> h_inter_id_to_ev_start_indices;
  core23::Tensor id_to_ev_start_indices;

  core23::DataType type;
  int max_ev_size;

  IntraModelReductionBufferAttr(std::shared_ptr<CoreResourceManager> core,
                                const EmbeddingCollectionParam &ebc_param, size_t grouped_id,
                                const std::vector<std::vector<int>> &h_lookup_ids_in_current_node);
};

struct IntraModelReductionBuffer {
  std::vector<core23::Tensor> data_list;  // num of nodes
  core23::Tensor data;

  const IntraModelReductionBufferAttr &attr;

  IntraModelReductionBuffer(std::shared_ptr<CoreResourceManager> core,
                            const IntraModelReductionBufferAttr &attr, int batch_size);
};

struct IntraModelForward {
  std::shared_ptr<CoreResourceManager> core_;

  void intra_forward(const core23::Tensor &evs, const core23::Tensor &bucket_range,
                     IntraModelCommBuffer &intra_model_comm_buffer, int batch_size);

  void dst_reduction(const IntraModelCommBuffer &intra_model_comm_buffer,
                     IntraModelReductionBuffer &reduction_buffer, int batch_size);
};
}  // namespace embedding
