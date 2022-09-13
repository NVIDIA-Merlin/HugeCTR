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

#include "HugeCTR/embedding/embedding_table.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "common.hpp"
namespace embedding {

// per gpu object
class IGroupedEmbeddingTable : public ILookup {
 public:
  virtual ~IGroupedEmbeddingTable() = default;

  virtual void update(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
                      size_t num_id_space_offset, const Tensor &id_space_list, Tensor &grad_ev,
                      const Tensor &grad_ev_offset) = 0;

  virtual void load(Tensor &keys, Tensor &id_space_offset, Tensor &embedding_table,
                    Tensor &ev_size_list, Tensor &id_space) = 0;

  virtual void dump(Tensor *keys, Tensor *id_space_offset, Tensor *embedding_table,
                    Tensor *ev_size_list, Tensor *id_space) = 0;

  virtual size_t size() const = 0;

  virtual size_t capacity() const = 0;

  virtual void clear() = 0;

  virtual void set_learning_rate(float lr) = 0;
};

class IDynamicEmbeddingTable : public IGroupedEmbeddingTable {
  virtual void evict(const Tensor &keys, size_t num_keys, const Tensor &id_space_offset,
                     size_t num_id_space_offset, const Tensor &id_space_list) = 0;
};

std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_embedding_table(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core_list,
    const EmbeddingCollectionParam &embedding_collection_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list,
    const std::vector<EmbeddingShardingParam> &emb_sharding_param_list);

std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_grouped_embedding_table(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::shared_ptr<CoreResourceManager> core,
    const EmbeddingCollectionParam &embedding_collection_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list);
}  // namespace embedding
