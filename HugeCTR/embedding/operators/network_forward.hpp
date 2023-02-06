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

#include "HugeCTR/core/buffer.hpp"
#include "HugeCTR/core/registry.hpp"
#include "HugeCTR/embedding/common.hpp"

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::Shape;
using core::Tensor;
using core::TensorList;

std::vector<size_t> cal_network_comm_buffer_size(
    int universal_batch_size, int num_gpus,
    const std::vector<std::vector<int>>& global_lookup_id_list,
    const std::vector<int>& ev_size_list);

struct NetworkIndices {
  Tensor network_ids;
  Tensor network_gpu_ids;
  Tensor network_offsets;
  Tensor network_dst_lookup_ids;

  void init(std::shared_ptr<CoreResourceManager> core,
            const std::vector<std::vector<int>>& h_global_lookup_ids);
};

struct NetworkBufferAttr : public EVBufferAttr {
  std::vector<Tensor> id_to_ev_size_list;
  TensorList id_to_ev_size;

  std::vector<Tensor> id_to_ev_start_indices_list;
  TensorList id_to_ev_start_indices;

  int num_gpus;
  std::vector<int> gpu_id_to_max_ev_elements;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam& ebc_param,
            size_t grouped_id, const std::vector<std::vector<int>>& h_global_lookup_ids);
};

struct NetworkBuffer {
  std::vector<Tensor> data_list;
  TensorList data;

  NetworkBufferAttr attr;

  void init(std::shared_ptr<CoreResourceManager> core, const NetworkBufferAttr& attr,
            int batch_size);
};

class NetworkForward {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;

 public:
  NetworkForward() = default;

  NetworkForward(std::shared_ptr<CoreResourceManager> core, int num_gpus);

  void compute(const Tensor& bucket_range, const NetworkBuffer& network_buffer,
               const NetworkIndices& network_indices, EmbeddingOutput& embedding_output,
               int batch_size);

  void compute(const TensorList& row_lengths, const Tensor& d_combiner_list,
               const TensorList& network_comm_buffer, const Tensor& network_ids,
               const Tensor& network_gpu_ids, const Tensor& network_offsets,
               const Tensor& network_dst_lookup_ids, const TensorList& network_ev_sizes,
               const TensorList& network_ev_offsets, TensorList& output_buffer,
               const Tensor& d_ev_size_offset, int batch_size, int max_ev_size);
};

}  // namespace embedding
