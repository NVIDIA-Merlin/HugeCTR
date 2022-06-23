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
#include "embedding_planner.hpp"

#include <nlohmann/json.hpp>
#include <unordered_set>

#include "embedding_collection.hpp"

namespace embedding {

EmbeddingPlanner::EmbeddingPlanner(const EmbeddingCollectionParam &param) : param_(param) {}

void EmbeddingPlanner::generate_embedding_plan_from_json_file(const std::string &plan_file) {
  nlohmann::json plan;
  std::ifstream plan_file_stream(plan_file);
  if (!plan_file_stream.is_open()) {
    HCTR_OWN_THROW(HugeCTR::Error_t::FileCannotOpen, plan_file + " can not open.");
  }
  plan_file_stream >> plan;
  plan_file_stream.close();

  int num_global_gpus = plan.size();
  global_embedding_sharding_param_list_.clear();
  global_embedding_sharding_param_list_.resize(num_global_gpus);

  auto check_and_find_in_json = [](const nlohmann::json &j, const std::string &key) {
    auto iter = j.find(key);
    if (iter == j.end()) {
      HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "No Key: " + key);
    }
    return *iter;
  };

  for (int global_id = 0; global_id < num_global_gpus; ++global_id) {
    auto local_plan = plan[global_id];
    for (auto &embedding_plan : local_plan) {
      auto local_embedding_list =
          check_and_find_in_json(embedding_plan, "local_embedding_list").get<std::vector<int>>();
      auto global_embedding_list = check_and_find_in_json(embedding_plan, "global_embedding_list")
                                       .get<std::vector<std::vector<int>>>();

      int sharding_id = 0;
      if (embedding_plan.find("sharding_id") != embedding_plan.end()) {
        sharding_id = (*embedding_plan.find("sharding_id")).get<int>();
      }

      int num_sharding = 1;
      if (embedding_plan.find("num_sharding") != embedding_plan.end()) {
        num_sharding = (*embedding_plan.find("num_sharding")).get<int>();
      }
      HCTR_CHECK_HINT(sharding_id < num_sharding,
                      "EmbeddingPlanner: illegal input. sharding_id %d >= num_sharding %d",
                      sharding_id, num_sharding);

      std::string table_placement_strategy =
          check_and_find_in_json(embedding_plan, "table_placement_strategy").get<std::string>();
      if (table_placement_strategy == "dp" && num_sharding > 1) {
        HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "dp embedding can not be sharded");
      }
      if (_table_placement_type_map.find(table_placement_strategy) ==
          _table_placement_type_map.end()) {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       table_placement_strategy + " is an unsupported table place type");
      }
      EmbeddingShardingParam sharding_param;
      sharding_param.local_embedding_list = local_embedding_list;
      sharding_param.global_embedding_list = global_embedding_list;
      sharding_param.sharding_id = sharding_id;
      sharding_param.num_sharding = num_sharding;
      sharding_param.table_placement_strategy =
          _table_placement_type_map.at(table_placement_strategy);

      global_embedding_sharding_param_list_[global_id].push_back(std::move(sharding_param));
    }
  }

  auto check_num_sharding_param_eqal_in_all_gpu =
      [this](const std::vector<EmbeddingShardingParam> &sharding_param_list) {
        return sharding_param_list.size() == global_embedding_sharding_param_list_[0].size();
      };
  HCTR_CHECK_HINT(std::all_of(global_embedding_sharding_param_list_.begin(),
                              global_embedding_sharding_param_list_.end(),
                              check_num_sharding_param_eqal_in_all_gpu),
                  "EmbeddingPlanner: illegal input. The num of embedding sharding param should be "
                  "the same on all gpu.");

  auto check_table_placement_strategy_eqal_in_all_gpu = [this, num_global_gpus] {
    int num_sharding_param = global_embedding_sharding_param_list_[0].size();
    for (int sharding_param_id = 0; sharding_param_id < num_sharding_param; ++sharding_param_id) {
      TablePlacementStrategy table_placement_strategy =
          global_embedding_sharding_param_list_[0][sharding_param_id].table_placement_strategy;
      for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
        HCTR_CHECK_HINT(table_placement_strategy ==
                            global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                                .table_placement_strategy,
                        "EmbeddingPlanner: illegal input. The table placement strategy is not "
                        "consistent between all gpus.");
      }
    }
  };
  check_table_placement_strategy_eqal_in_all_gpu();

  auto check_local_global_embedding_list = [this, num_global_gpus] {
    for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
      int num_sharding_param = global_embedding_sharding_param_list_[gpu_id].size();
      for (int sharding_param_id = 0; sharding_param_id < num_sharding_param; ++sharding_param_id) {
        HCTR_CHECK_HINT(
            global_embedding_sharding_param_list_[gpu_id][sharding_param_id].local_embedding_list ==
                global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                    .global_embedding_list[gpu_id],
            "EmbeddingPlanner: illegal input. The local_embedding_list is not compatible with "
            "global_embedding_list");

        size_t num_embedding = global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                                   .local_embedding_list.size();
        std::vector<int> unique_local_embedding_list =
            global_embedding_sharding_param_list_[gpu_id][sharding_param_id].local_embedding_list;

        std::sort(unique_local_embedding_list.begin(), unique_local_embedding_list.end());
        auto last =
            std::unique(unique_local_embedding_list.begin(), unique_local_embedding_list.end());
        unique_local_embedding_list.erase(last, unique_local_embedding_list.end());

        HCTR_CHECK_HINT(num_embedding == unique_local_embedding_list.size(),
                        "EmbeddingPlanner: illegal input. The local_embedding_list should not have "
                        "duplicate embedding");
      }
    }

    int num_sharding_param = global_embedding_sharding_param_list_[0].size();
    for (int sharding_param_id = 0; sharding_param_id < num_sharding_param; ++sharding_param_id) {
      for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
        HCTR_CHECK_HINT(
            global_embedding_sharding_param_list_[0][sharding_param_id].global_embedding_list ==
                global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                    .global_embedding_list,
            "EmbeddingPlanner: illegal input. The global_embedding_list is not consistent between "
            "all gpu on same embedding");
      }
    }
  };
  check_local_global_embedding_list();

  auto check_dp_embedding = [this, num_global_gpus] {
    int num_sharding_param = global_embedding_sharding_param_list_[0].size();
    for (int sharding_param_id = 0; sharding_param_id < num_sharding_param; ++sharding_param_id) {
      std::set<std::vector<int>> embedding_set;
      for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
        if (global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                .table_placement_strategy == TablePlacementStrategy::ModelParallel) {
          continue;
        }
        if (gpu_id == 0) {
          embedding_set.insert(global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                                   .local_embedding_list);
        }
        HCTR_CHECK_HINT(
            embedding_set.find(global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                                   .local_embedding_list) != embedding_set.end(),
            "EmbeddingPlanner: illegal input. DataParallel embedding should have same "
            "local_embedding_list.");
      }
    }
  };
  check_dp_embedding();

  auto check_mp_embeddding_sharding = [this, num_global_gpus] {
    int num_sharding_param = global_embedding_sharding_param_list_[0].size();
    for (int sharding_param_id = 0; sharding_param_id < num_sharding_param; ++sharding_param_id) {
      std::map<std::vector<int>, std::vector<int>> embedding_and_sharding_id_dict;
      std::map<std::vector<int>, int> embedding_and_num_sharding_dict;
      for (int gpu_id = 0; gpu_id < num_global_gpus; ++gpu_id) {
        auto &embedding =
            global_embedding_sharding_param_list_[gpu_id][sharding_param_id].local_embedding_list;
        if (global_embedding_sharding_param_list_[gpu_id][sharding_param_id]
                .table_placement_strategy == TablePlacementStrategy::DataParallel) {
          continue;
        }
        embedding_and_sharding_id_dict[embedding].push_back(
            global_embedding_sharding_param_list_[gpu_id][sharding_param_id].sharding_id);
        if (embedding_and_num_sharding_dict.find(embedding) ==
            embedding_and_num_sharding_dict.end()) {
          embedding_and_num_sharding_dict[embedding] =
              global_embedding_sharding_param_list_[gpu_id][sharding_param_id].num_sharding;
        }
        HCTR_CHECK_HINT(
            embedding_and_num_sharding_dict[embedding] ==
                global_embedding_sharding_param_list_[gpu_id][sharding_param_id].num_sharding,
            "EmbeddingPlanner: illegal input. The num_sharding is not consistent between all "
            "sharding item.");
      }
      std::unordered_set<int> unique_embedding;
      for (const auto &[embedding_list, num_sharding] : embedding_and_num_sharding_dict) {
        for (int embedding : embedding_list) {
          HCTR_CHECK_HINT(unique_embedding.find(embedding) == unique_embedding.end(),
                          "EmbeddingPlanner: illegal input. Different sharding embedding list have "
                          "same embedding.");
          unique_embedding.insert(embedding);
        }
      }
      for (const auto &[embedding_list, num_sharding] : embedding_and_num_sharding_dict) {
        auto sharding_id_list = embedding_and_sharding_id_dict[embedding_list];

        HCTR_CHECK_HINT(sharding_id_list.size() == static_cast<size_t>(num_sharding),
                        "EmbeddingPlanner: illegal input. size of sharding id list is not "
                        "consistent with num_sharding.");

        std::sort(sharding_id_list.begin(), sharding_id_list.end());
        for (size_t i = 0; i < sharding_id_list.size(); ++i) {
          HCTR_CHECK_HINT(i == static_cast<size_t>(sharding_id_list[i]),
                          "EmbeddingPlanner: illegal input. Miss sharding id.");
        }
      }
    }
  };
  check_mp_embeddding_sharding();
}

void EmbeddingPlanner::generate_embedding_plan(const std::string &strategy) {}

std::unique_ptr<IEmbeddingCollectionForward> EmbeddingPlanner::create_embedding_collection_forward(
    std::shared_ptr<CoreResourceManager> core) {
  int global_id = core->get_global_gpu_id();

  return std::make_unique<EmbeddingCollectionForward>(
      core, param_, global_embedding_sharding_param_list_[global_id]);
}

std::unique_ptr<IEmbeddingCollectionBackward>
EmbeddingPlanner::create_embedding_collection_backward(std::shared_ptr<CoreResourceManager> core) {
  int global_id = core->get_global_gpu_id();

  return std::make_unique<EmbeddingCollectionBackward>(
      core, param_, global_embedding_sharding_param_list_[global_id]);
}
}  // namespace embedding
