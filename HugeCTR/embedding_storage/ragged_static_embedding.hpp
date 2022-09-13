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
#include "HugeCTR/core/registry.hpp"
#include "embedding_table.hpp"
namespace embedding {
using HugeCTR::CudaDeviceContext;

class RaggedStaticEmbeddingTable final : public IGroupedEmbeddingTable {
  std::shared_ptr<CoreResourceManager> core_;
  Tensor table_ids_;
  size_t emb_table_size_;

  Tensor keys_;
  Tensor num_key_per_table_offset_;

  Tensor emb_table_;
  Tensor emb_table_ev_offset_;  // num_local_id_space + 1
  Tensor local_ev_size_list_;   // num_local_id_space

  HugeCTR::OptParams opt_param_;

 public:
  RaggedStaticEmbeddingTable(const HugeCTR::GPUResource &gpu_resource,
                             std::shared_ptr<CoreResourceManager> core,
                             const std::vector<EmbeddingTableParam> &global_emb_table_param_list,
                             const EmbeddingCollectionParam &ebc_param,
                             const EmbeddingShardingParam &sharding_param,
                             const HugeCTR::OptParams &opt_param);

  RaggedStaticEmbeddingTable(const HugeCTR::GPUResource &gpu_resource,
                             std::shared_ptr<CoreResourceManager> core,
                             const std::vector<EmbeddingTableParam> &global_emb_table_param_list,
                             const EmbeddingCollectionParam &ebc_param,
                             const EmbeddingShardParam &shard_param,
                             const HugeCTR::OptParams &opt_param);

  RaggedStaticEmbeddingTable(const HugeCTR::GPUResource &gpu_resource,
                             std::shared_ptr<CoreResourceManager> core,
                             const std::vector<EmbeddingTableParam> &table_params,
                             const EmbeddingCollectionParam &ebc_param, size_t emb_id,
                             const HugeCTR::OptParams &opt_param);
  // void hash_insert(const Tensor &keys, size_t num_keys, const Tensor &offsets, size_t
  // num_offsets, const Tensor &d_id_space_list, Tensor &indices) override;

  void lookup(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
              size_t num_id_space_offset, const Tensor &id_space,
              TensorList &embedding_vec) override;

  void update(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
              size_t num_id_space_offset, const Tensor &id_space_list, Tensor &grad_ev,
              const Tensor &grad_ev_offset) override;

  void load(Tensor &keys, Tensor &id_space_offset, Tensor &embedding_table, Tensor &ev_size_list,
            Tensor &id_space) override;

  void dump(Tensor *keys, Tensor *id_space_offset, Tensor *embedding_table, Tensor *ev_size_list,
            Tensor *id_space) override;

  size_t size() const override;

  size_t capacity() const override;

  void clear() override;

  void set_learning_rate(float lr) override { opt_param_.lr = lr; }
};
}  // namespace embedding
