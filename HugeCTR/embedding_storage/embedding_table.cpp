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

#include "dynamic_embedding.hpp"
#include "ragged_static_embedding.hpp"
namespace embedding {

std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_embedding_table(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::vector<std::shared_ptr<CoreResourceManager>> core_list,
    const EmbeddingCollectionParam &emb_collection_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list,
    const std::vector<EmbeddingShardingParam> &emb_sharding_param_list) {
  int num_local_gpu = core_list.size();
  auto check_id_space = [&] {
    for (size_t i = 0; i < emb_table_param_list.size(); ++i) {
      const EmbeddingTableParam &emb_table_param = emb_table_param_list[i];
      HCTR_CHECK_HINT(emb_table_param.table_id == static_cast<int>(i),
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

  auto is_dynamic_embedding_table = [&](const EmbeddingShardingParam &local_sharding_param) {
    if (local_sharding_param.local_embedding_list.size() == 0) {
      return false;
    }
    for (int embedding_id : local_sharding_param.local_embedding_list) {
      int id_space = emb_collection_param.embedding_params[embedding_id].id_space;
      if (emb_table_param_list[id_space].max_vocabulary_size >= 0) {
        return false;
      }
    }
    return true;
  };

  auto get_opt_params = [&](const EmbeddingShardingParam &local_sharding_param) {
    if (local_sharding_param.local_embedding_list.size() == 0) {
      HugeCTR::OptParams empty;
      return empty;
    }
    int first_id_space = emb_collection_param.embedding_params[0].id_space;
    return emb_table_param_list[first_id_space].opt_param;
  };
  check_id_space();
  check_optimizer();

  std::vector<std::unique_ptr<IGroupedEmbeddingTable>> embedding_table_list;
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpu; ++local_gpu_id) {
    auto core = core_list[local_gpu_id];
    int global_gpu_id = core->get_global_gpu_id();
    const EmbeddingShardingParam &local_sharding_param = emb_sharding_param_list[global_gpu_id];

    if (is_dynamic_embedding_table(local_sharding_param)) {
      embedding_table_list.push_back(std::make_unique<DynamicEmbeddingTable>(
          core, emb_table_param_list, emb_collection_param, local_sharding_param));
    } else {
      embedding_table_list.push_back(std::make_unique<RaggedStaticEmbeddingTable>(
          *resource_manager->get_local_gpu(local_gpu_id), core, emb_table_param_list,
          emb_collection_param, local_sharding_param, get_opt_params(local_sharding_param)));
    }
  }
  return embedding_table_list;
}

std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_grouped_embedding_table(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list) {
  int local_gpu_id = core->get_local_gpu_id();

  // auto check_optimizer = [&] {
  //   for (int local_gpu_id = 0; local_gpu_id < num_local_gpu; ++local_gpu_id) {
  //     auto core = core_list[local_gpu_id];
  //     int global_gpu_id = core->get_global_gpu_id();
  //     const EmbeddingShardingParam &local_sharding_param =
  //     emb_sharding_param_list[global_gpu_id];

  //     for (int embedding_id : local_sharding_param.local_embedding_list) {
  //       int id_space = emb_collection_param.embedding_params[embedding_id].id_space;
  //       const auto &opt_param = emb_table_param_list[id_space].opt_param;
  //       int first_id_space = emb_collection_param.embedding_params[0].id_space;
  //       const auto &first_opt_param = emb_table_param_list[first_id_space].opt_param;

  //       HCTR_CHECK_HINT(
  //           opt_param == first_opt_param,
  //           "create_embedding_storage failed! only support sharding table with same optimizer.");
  //     }
  //   }
  // };

  auto is_dynamic_embedding_table = [&](size_t emb_id) {
    const auto &table_ids = ebc_param.emb_params[emb_id].table_ids;
    if (table_ids.size() == 0) {
      return false;
    }
    for (int table_id : table_ids) {
      if (emb_table_param_list[table_id].max_vocabulary_size < 0) {
        return true;
      }
    }
    return false;
  };

  auto get_opt_params = [&](size_t emb_id) {
    const auto &table_ids = ebc_param.emb_params[emb_id].table_ids;
    if (table_ids.size() == 0) {
      HugeCTR::OptParams empty;
      return empty;
    }
    int first_table_id = table_ids[0];
    return emb_table_param_list[first_table_id].opt_param;
  };

  // check_optimizer();

  std::vector<std::unique_ptr<IGroupedEmbeddingTable>> embedding_table_list;
  for (size_t emb_id = 0; emb_id < ebc_param.emb_params.size(); ++emb_id) {
    if (is_dynamic_embedding_table(emb_id)) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "dynamic embedding table not supported.");
      //  embedding_table_list.push_back(std::make_unique<DynamicEmbeddingTable>(
      //     core, emb_table_param_list, emb_collection_param, local_sharding_param));
    } else {
      embedding_table_list.push_back(std::make_unique<RaggedStaticEmbeddingTable>(
          *resource_manager->get_local_gpu(local_gpu_id), core, emb_table_param_list, ebc_param,
          emb_id, get_opt_params(emb_id)));
    }
  }
  return embedding_table_list;
}

}  // namespace embedding