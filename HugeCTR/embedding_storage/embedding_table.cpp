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
#include "embedding_table.hpp"

#include <cassert>

#include "ragged_static_embedding.hpp"
namespace embedding {

std::vector<std::unique_ptr<IEmbeddingTable>> create_embedding_table(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core_list,
    const EmbeddingCollectionParam &emb_collection_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list,
    const std::vector<EmbeddingShardingParam> &emb_sharding_param_list) {
  int num_local_gpu = core_list.size();
  auto check_id_space = [&] {
    for (size_t i = 0; i < emb_table_param_list.size(); ++i) {
      const EmbeddingTableParam &emb_table_param = emb_table_param_list[i];
      HCTR_CHECK_HINT(emb_table_param.id_space == static_cast<int>(i),
                      "create_embedding_table failed! embedding table id space should be sorted "
                      "and continous.");
    }
    int max_id_space = emb_table_param_list.size();

    for (int local_gpu_id = 0; local_gpu_id < num_local_gpu; ++local_gpu_id) {
      auto core = core_list[local_gpu_id];
      int global_gpu_id = core->get_global_gpu_id();
      const EmbeddingShardingParam &local_sharding_param = emb_sharding_param_list[global_gpu_id];
      for (int embedding_id : local_sharding_param.local_embedding_list) {
        int id_space = emb_collection_param.embedding_params[embedding_id].id_space;
        HCTR_CHECK_HINT(id_space < max_id_space,
                        "create_embedding_table failed! embedding id_space is not match with "
                        "embedding table id_space.");
      }
    }
  };

  check_id_space();

  auto check_optimizer = [&] {
    for (int local_gpu_id = 0; local_gpu_id < num_local_gpu; ++local_gpu_id) {
      auto core = core_list[local_gpu_id];
      int global_gpu_id = core->get_global_gpu_id();
      const EmbeddingShardingParam &local_sharding_param = emb_sharding_param_list[global_gpu_id];

      for (int embedding_id : local_sharding_param.local_embedding_list) {
        int id_space = emb_collection_param.embedding_params[embedding_id].id_space;
        const auto &opt_param = emb_table_param_list[id_space].opt_param;
        int first_id_space = emb_collection_param.embedding_params[0].id_space;
        const auto &first_opt_param = emb_table_param_list[first_id_space].opt_param;

        HCTR_CHECK_HINT(
            opt_param == first_opt_param,
            "create_embedding_table failed! only support sharding table with same optimizer.");
      }
    }
  };
  check_optimizer();

  auto get_opt_params = [&](const EmbeddingShardingParam &local_sharding_param) {
    if (local_sharding_param.local_embedding_list.size() == 0) {
      HugeCTR::OptParams empty;
      return empty;
    }
    int first_id_space = emb_collection_param.embedding_params[0].id_space;
    return emb_table_param_list[first_id_space].opt_param;
  };

  std::vector<std::unique_ptr<IEmbeddingTable>> embedding_table_list;
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpu; ++local_gpu_id) {
    auto core = core_list[local_gpu_id];
    int global_gpu_id = core->get_global_gpu_id();
    const EmbeddingShardingParam &local_sharding_param = emb_sharding_param_list[global_gpu_id];

    embedding_table_list.push_back(std::make_unique<RaggedStaticEmbeddingTable>(
        *resource_manager->get_local_gpu(local_gpu_id), core, emb_table_param_list,
        emb_collection_param, local_sharding_param, get_opt_params(local_sharding_param)));
    // TODO: add support for dynamic embedding table
  }
  return embedding_table_list;
}

}  // namespace embedding