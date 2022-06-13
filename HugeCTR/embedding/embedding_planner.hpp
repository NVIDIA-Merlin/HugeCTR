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
#include "embedding.hpp"

namespace embedding {
using core::CoreResourceManager;

class EmbeddingPlanner {
  EmbeddingCollectionParam param_;
  std::vector<std::vector<EmbeddingShardingParam>> global_embedding_sharding_param_list_;
  
 public:
  EmbeddingPlanner(const EmbeddingCollectionParam &param);

  const std::vector<std::vector<EmbeddingShardingParam>> &get_gpu_major_global_embedding_sharding_param_list() const {
    return global_embedding_sharding_param_list_;
  }

  std::vector<std::vector<EmbeddingShardingParam>> get_table_major_global_embedding_sharding_param_list() const {
    std::vector<std::vector<EmbeddingShardingParam>> table_major_param_list;
    size_t num_table = global_embedding_sharding_param_list_[0].size();
    table_major_param_list.resize(num_table);
    for (size_t table_id = 0; table_id < num_table; ++table_id) {
      for (size_t gpu_id = 0; gpu_id < global_embedding_sharding_param_list_.size(); ++gpu_id) {
        table_major_param_list[table_id].push_back(global_embedding_sharding_param_list_[gpu_id][table_id]);
      }
    }
    return table_major_param_list;
  }

  void generate_embedding_plan_from_json_file(const std::string &plan_file);

  void generate_embedding_plan(const std::string &strategy);
   
  std::unique_ptr<IEmbeddingCollectionForward> create_embedding_collection_forward(std::shared_ptr<CoreResourceManager> core);
  
  std::unique_ptr<IEmbeddingCollectionBackward> create_embedding_collection_backward(std::shared_ptr<CoreResourceManager> core);
};
}  // namespace embedding
