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

#include <core23/registry.hpp>
#include <embedding/common.hpp>

namespace embedding {

class DPModelForward {
  std::shared_ptr<CoreResourceManager> core_;

 public:
  DPModelForward() = default;

  DPModelForward(std::shared_ptr<CoreResourceManager> core);

  void sparse_forward(const core23::Tensor &lookup_res, const core23::Tensor &dp_bucket_range,
                      const core23::Tensor &local_lookup_ids, EmbeddingOutput &embedding_output,
                      int batch_size_per_gpu);
};

struct ModelCommBufferAttr {
  std::vector<int> h_id_to_ev_size;
  core23::Tensor id_to_ev_size;

  std::vector<int> h_id_to_ev_start_indices;
  core23::Tensor id_to_ev_start_indices;

  int num_lookup;
  int num_gpus;
  int max_ev_elements;

  EmbeddingLayout layout;
  int max_ev_size;
  core23::DataType type;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id);
};

struct ModelCommBuffer {
  std::vector<core23::Tensor> data_list;
  core23::Tensor data;

  ModelCommBufferAttr attr;

  void init(std::shared_ptr<CoreResourceManager> core, const ModelCommBufferAttr &attr,
            int batch_size);

  void init_from_device_buffer(std::shared_ptr<CoreResourceManager> core,
                               const std::vector<core23::Tensor> &data_buffer_list,
                               const ModelCommBufferAttr &attr);
};

struct DenseModelCommBufferAttr {
  int num_local_lookup;
  int num_gpus;
  int max_hotness_sum;

  EmbeddingLayout layout;
  int ev_size;
  core23::DataType type;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id);
};

struct DenseModelCommBuffer {
  core23::Tensor data;

  DenseModelCommBufferAttr attr;

  void init(std::shared_ptr<CoreResourceManager> core, const DenseModelCommBufferAttr &attr,
            int batch_size);
};

struct ModelForward {
  std::shared_ptr<CoreResourceManager> core_;

  void sparse_forward(const core23::Tensor &mp_ev, const core23::Tensor &bucket_range,
                      ModelCommBuffer &model_comm_buffer, int batch_size);
  void dense_forward(const core23::Tensor &mp_ev, const core23::Tensor &reverse_idx,
                     DenseModelCommBuffer &model_comm_buffer, int batch_size, size_t num_key);
};

}  // namespace embedding
