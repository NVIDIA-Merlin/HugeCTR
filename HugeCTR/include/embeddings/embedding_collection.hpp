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

#include <HugeCTR/include/optimizer.hpp>
#include <embedding/common.hpp>
#include <embedding/data_distributor/data_distributor.hpp>
#include <embedding/embedding.hpp>
#include <embedding/gpu_barrier/gpu_barrier.hpp>
#include <embedding/operators/transpose_input.hpp>
#include <embedding_storage/embedding_table.hpp>
#include <optimizer.hpp>

namespace HugeCTR {

class EmbeddingTableConfig {
 public:
  std::string name;
  ::embedding::EmbeddingTableParam table_param;

  EmbeddingTableConfig() {}

  EmbeddingTableConfig(const std::string &name, int max_vocabulary_size, int ev_size,
                       std::optional<HugeCTR::OptParams> opt_param_or_empty,
                       std::optional<::embedding::InitParams> init_param_or_empty)
      : name(name) {
    HugeCTR::OptParams opt_param;
    if (opt_param_or_empty.has_value()) {
      opt_param = opt_param_or_empty.value();
    } else {
      opt_param.optimizer = HugeCTR::Optimizer_t::NOT_INITIALIZED;
    }

    ::embedding::InitParams init_param{ev_size};
    if (init_param_or_empty.has_value()) {
      init_param = init_param_or_empty.value();
    }

    this->table_param =
        ::embedding::EmbeddingTableParam{-1, max_vocabulary_size, ev_size, opt_param, init_param};
  }
};

using ShardStrategy = std::tuple<std::string, std::vector<std::string>>;

inline std::string get_table_place_strategy(const ShardStrategy &s) { return std::get<0>(s); }

inline std::vector<std::string> get_table_group_strategy(const ShardStrategy &s) {
  return std::get<1>(s);
}

class EmbeddingCollectionConfig {
 public:
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;

  using LookupConfig = std::pair<std::string, ::embedding::LookupParam>;
  std::vector<LookupConfig> lookup_configs_;

  std::vector<EmbeddingTableConfig> emb_table_config_list_;

  std::vector<ShardStrategy> shard_strategy_;
  std::vector<std::vector<std::string>> shard_matrix_;

  ::embedding::EmbeddingLayout output_layout_;

  ::embedding::SortStrategy sort_strategy_;
  ::embedding::KeysPreprocessStrategy keys_preprocess_strategy_;
  ::embedding::AllreduceStrategy allreduce_strategy_;
  ::embedding::CommunicationStrategy comm_strategy_;

  std::string batch_major_output_name_;

  // if we need more configuration about EmbeddingCollection
  EmbeddingCollectionConfig(bool use_exclusive_keys,
                            ::embedding::CommunicationStrategy comm_strategy)
      : output_layout_(::embedding::EmbeddingLayout::FeatureMajor),
        sort_strategy_(use_exclusive_keys ? ::embedding::SortStrategy::Radix
                                          : ::embedding::SortStrategy::Segmented),
        keys_preprocess_strategy_(::embedding::KeysPreprocessStrategy::AddOffset),
        allreduce_strategy_(::embedding::AllreduceStrategy::Dense),
        comm_strategy_(comm_strategy) {
    if (comm_strategy_ == ::embedding::CommunicationStrategy::Hierarchical) {
      HCTR_LOG(INFO, ROOT, "Using Hier Communication Strategy\n");
    }
  }

  void embedding_lookup(const EmbeddingTableConfig &emb_table_config,
                        const std::string &bottom_name, const std::string &top_name,
                        const std::string &combiner_str) {
    ::embedding::Combiner combiner;
    if (combiner_str == "concat") {
      combiner = ::embedding::Combiner::Concat;
    } else if (combiner_str == "sum") {
      combiner = ::embedding::Combiner::Sum;
    } else if (combiner_str == "average") {
      combiner = ::embedding::Combiner::Average;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, combiner_str + " is not supported.");
    }

    ::embedding::LookupParam lookup_param{static_cast<int>(lookup_configs_.size()), -1, combiner,
                                          -1, emb_table_config.table_param.ev_size};
    lookup_configs_.push_back({emb_table_config.name, lookup_param});
    bottom_names_.push_back(bottom_name);
    top_names_.push_back(top_name);

    // Make sure the order of emb table config will not changed after insert
    bool existed = false;
    for (auto &existed_emb_table_config : emb_table_config_list_) {
      if (existed_emb_table_config.name == emb_table_config.name) {
        existed = true;
      }
    }
    if (existed) return;
    if (emb_table_config.table_param.max_vocabulary_size < 0) {
      keys_preprocess_strategy_ = ::embedding::KeysPreprocessStrategy::None;
      allreduce_strategy_ = ::embedding::AllreduceStrategy::Sparse;
    }
    emb_table_config_list_.push_back(emb_table_config);
  }

  void embedding_lookup(const std::vector<EmbeddingTableConfig> &emb_table_config,
                        const std::vector<std::string> &bottom_name, const std::string &top_name,
                        const std::vector<std::string> &combiner_str) {
    HCTR_CHECK_HINT(lookup_configs_.empty(), "empty lookup params required");
    output_layout_ = ::embedding::EmbeddingLayout::BatchMajor;
    batch_major_output_name_ = top_name;

    HCTR_CHECK(emb_table_config.size() == bottom_name.size());
    HCTR_CHECK(emb_table_config.size() == combiner_str.size());

    for (size_t i = 0; i < emb_table_config.size(); ++i) {
      embedding_lookup(emb_table_config[i], bottom_name[i], top_name + std::to_string(i),
                       combiner_str[i]);
    }
  }

  void shard(const std::vector<std::vector<std::string>> &shard_matrix,
             const std::vector<ShardStrategy> &shard_strategy) {
    shard_matrix_.clear();
    shard_strategy_.clear();

    shard_matrix_ = shard_matrix;
    shard_strategy_ = shard_strategy;
  }
};

using TableNameToIDDict = std::unordered_map<std::string, int>;
inline TableNameToIDDict create_table_name_to_id_dict_from_ebc_config(
    const EmbeddingCollectionConfig &config) {
  TableNameToIDDict table_name_to_id_dict;
  int table_id = 0;
  for (auto &c : config.emb_table_config_list_) {
    table_name_to_id_dict[c.name] = table_id;
    table_id += 1;
  }
  return table_name_to_id_dict;
}

inline void check_table_name_correct(
    const std::unordered_map<std::string, std::pair<int, int>> &ebc_name_to_id,
    const std::vector<std::string> &table_names) {
  // check table names is right
  for (auto &name : table_names) {
    if (ebc_name_to_id.find(name) == ebc_name_to_id.end()) {
      HCTR_CHECK_HINT(ebc_name_to_id.find(name) == ebc_name_to_id.end(),
                      "embedding_load can't find table name : %s in model\n", name.c_str());
    }
  }
}

inline std::vector<::embedding::LookupParam> create_lookup_params_from_ebc_config(
    const TableNameToIDDict &table_name_to_id_dict, const EmbeddingCollectionConfig &config) {
  std::vector<::embedding::LookupParam> lookup_params;
  for (auto &lookup_config : config.lookup_configs_) {
    const auto &name = lookup_config.first;
    auto lookup_param = lookup_config.second;
    HCTR_CHECK_HINT(table_name_to_id_dict.find(name) != table_name_to_id_dict.end(),
                    "create_lookup_params_from_ebc_config error, no such name: %s\n", name.c_str());
    lookup_param.table_id = table_name_to_id_dict.at(name);
    lookup_params.push_back(lookup_param);
  }
  return lookup_params;
};

inline std::vector<::embedding::EmbeddingTableParam> create_table_params_from_ebc_config(
    const TableNameToIDDict &table_name_to_id_dict, const EmbeddingCollectionConfig &config) {
  std::vector<::embedding::EmbeddingTableParam> table_params;
  for (auto &table_config : config.emb_table_config_list_) {
    const auto &name = table_config.name;
    auto emb_table_param = table_config.table_param;
    HCTR_CHECK_HINT(table_name_to_id_dict.find(name) != table_name_to_id_dict.end(),
                    "create_table_params_from_ebc_config error, no such name: %s\n", name.c_str());
    emb_table_param.table_id = table_name_to_id_dict.at(name);
    table_params.push_back(emb_table_param);
  }
  return table_params;
};

inline std::vector<std::vector<int>> create_shard_matrix_from_ebc_config(
    const TableNameToIDDict &table_name_to_id_dict, const EmbeddingCollectionConfig &config) {
  std::vector<std::vector<int>> shard_matrix;
  int num_gpus = static_cast<int>(config.shard_matrix_.size());
  shard_matrix.resize(num_gpus);
  int num_table = static_cast<int>(table_name_to_id_dict.size());
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    shard_matrix[gpu_id].assign(num_table, 0);
  }

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    const auto &shard_on_each_gpu = config.shard_matrix_[gpu_id];
    for (auto &name : shard_on_each_gpu) {
      HCTR_CHECK_HINT(table_name_to_id_dict.find(name) != table_name_to_id_dict.end(),
                      "create_shard_matrix_from_ebc_config error, no such name: %s\n",
                      name.c_str());
      HCTR_CHECK_HINT(table_name_to_id_dict.at(name) < num_table,
                      "create_shard_matrix_from_ebc_config error, name is out of range: %s\n",
                      name.c_str());
      shard_matrix[gpu_id][table_name_to_id_dict.at(name)] = 1;
    }
  }
  return shard_matrix;
}

inline std::vector<::embedding::GroupedEmbeddingParam>
create_grouped_embedding_param_from_ebc_config(const TableNameToIDDict &table_name_to_id_dict,
                                               const EmbeddingCollectionConfig &config) {
  std::vector<::embedding::GroupedEmbeddingParam> grouped_embedding_params;
  for (auto &shard_strategy : config.shard_strategy_) {
    auto placement_strategy_string = get_table_place_strategy(shard_strategy);
    ::embedding::TablePlacementStrategy placement_strategy;
    if (placement_strategy_string == "mp") {
      placement_strategy = ::embedding::TablePlacementStrategy::ModelParallel;
    } else if (placement_strategy_string == "dp") {
      placement_strategy = ::embedding::TablePlacementStrategy::DataParallel;
    } else {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "table placement strategy is not match");
    }

    std::vector<int> table_ids;
    auto group_strategy = get_table_group_strategy(shard_strategy);
    for (auto &table_name : group_strategy) {
      HCTR_CHECK_HINT(table_name_to_id_dict.find(table_name) != table_name_to_id_dict.end(),
                      "create_grouped_embedding_param_from_ebc_config error, no such name: %s\n",
                      table_name.c_str());
      table_ids.push_back(table_name_to_id_dict.at(table_name));
    }
    // require ordered
    std::sort(table_ids.begin(), table_ids.end());
    ::embedding::GroupedEmbeddingParam grouped_emb_param{placement_strategy, table_ids};
    grouped_embedding_params.push_back(std::move(grouped_emb_param));
  }
  return grouped_embedding_params;
}

}  // namespace HugeCTR

namespace embedding {

class EmbeddingCollection {
 private:
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> embeddings_, eval_embeddings_;

  std::vector<std::vector<EmbeddingOutputAttr>> embedding_output_attrs;
  std::vector<std::vector<Wgrad>> wgrad_list_;
  std::unique_ptr<HugeCTR::GPUBarrier> gpu_barrier_;

 public:
  // Fix:load and dump use these , put it on public temporary
  std::vector<HugeCTR::OptParams> embedding_optimizers_;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingTable>>> embedding_tables_;
  EmbeddingCollectionParam ebc_param_;
  EmbeddingCollectionParam eval_ebc_param_;

 public:
  EmbeddingCollection(std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
                      std::vector<std::shared_ptr<CoreResourceManager>> core,
                      const EmbeddingCollectionParam &ebc_param,
                      const EmbeddingCollectionParam &eval_ebc_param,
                      const std::vector<EmbeddingTableParam> &emb_table_param_list);

  void forward_per_gpu(bool is_train, int gpu_id, const HugeCTR::DataDistributor::Result &input,
                       core23::Tensor &output_buffer, int batch_size);

  void backward_per_gpu(int gpu_id, const HugeCTR::DataDistributor::Result &input,
                        const core23::Tensor &top_grad, int batch_size);

  void update_per_gpu(int gpu_id);

  void set_learning_rate(float lr);

  std::vector<std::vector<IGroupedEmbeddingTable *>> get_grouped_embedding_tables() {
    std::vector<std::vector<IGroupedEmbeddingTable *>> grouped_embedding_tables;
    grouped_embedding_tables.resize(embedding_tables_.size());
    for (size_t i = 0; i < embedding_tables_.size(); ++i) {
      for (size_t j = 0; j < embedding_tables_[i].size(); ++j) {
        grouped_embedding_tables[i].push_back(embedding_tables_[i][j].get());
      }
    }
    return grouped_embedding_tables;
  }
};

}  // namespace embedding
