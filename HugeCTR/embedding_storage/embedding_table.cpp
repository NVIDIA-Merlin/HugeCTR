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
std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_grouped_embedding_tables(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list) {
  int local_gpu_id = core->get_local_gpu_id();

  auto is_dynamic_embedding_table = [&](const std::vector<int> &table_ids) {
    for (int table_id : table_ids) {
      if (emb_table_param_list[table_id].max_vocabulary_size < 0) {
        return true;
      }
    }
    return false;
  };

  auto get_opt_params = [&](const std::vector<int> &table_ids) {
    int first_table_id = table_ids[0];
    for (int table_id : table_ids) {
      if (emb_table_param_list[table_id].opt_param !=
          emb_table_param_list[first_table_id].opt_param) {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "grouped embedding table does not support grouping embedding table with "
                       "different optimizer.");
      }
    }
    return emb_table_param_list[first_table_id].opt_param;
  };

  // check_optimizer();

  std::vector<std::unique_ptr<IGroupedEmbeddingTable>> embedding_table_list;
  for (size_t grouped_id = 0; grouped_id < ebc_param.grouped_emb_params.size(); ++grouped_id) {
    const auto &table_ids = ebc_param.grouped_emb_params[grouped_id].table_ids;
    if (table_ids.size() == 0) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                     "grouped embedding table does not support empty grouped ids.");
    }
    HugeCTR::OptParams opt_params = get_opt_params(table_ids);

    if (is_dynamic_embedding_table(table_ids)) {
      // ebc_param.is_dynamic = true;
      embedding_table_list.push_back(std::make_unique<DynamicEmbeddingTable>(
          *resource_manager->get_local_gpu(local_gpu_id), core, emb_table_param_list, ebc_param,
          grouped_id, opt_params));
    } else {
      embedding_table_list.push_back(std::make_unique<RaggedStaticEmbeddingTable>(
          *resource_manager->get_local_gpu(local_gpu_id), core, emb_table_param_list, ebc_param,
          grouped_id, opt_params));
    }
  }
  return embedding_table_list;
}

}  // namespace embedding
