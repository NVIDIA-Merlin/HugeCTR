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
#include <embedding_storage/ragged_static_embedding.hpp>
#include <include/exchange_wgrad.hpp>
#include <include/network_buffer_channels.hpp>
#include <optimizer.hpp>
#include <tensor2.hpp>

namespace HugeCTR {

class EmbeddingTableConfig {
 public:
  std::string name;
  ::embedding::EmbeddingTableParam table_param;

  EmbeddingTableConfig() {}

  EmbeddingTableConfig(const std::string &name, int64_t max_vocabulary_size, int ev_size,
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

using TableVariant = std::variant<std::string, std::tuple<std::string, int>>;
using ShardStrategy = std::tuple<std::string, std::vector<TableVariant>>;

inline std::string get_table_place_strategy(const ShardStrategy &s) { return std::get<0>(s); }

inline std::vector<TableVariant> get_table_group_strategy(const ShardStrategy &s) {
  return std::get<1>(s);
}

inline std::string get_table_name(TableVariant v) {
  std::string table_name;
  if (std::string *name_ptr = std::get_if<std::string>(&v)) {
    table_name = *name_ptr;
  } else if (std::tuple<std::string, int> *tuple_ptr =
                 std::get_if<std::tuple<std::string, int>>(&v)) {
    table_name = std::get<0>(*tuple_ptr);
  } else {
    HCTR_OWN_THROW(Error_t::IllegalCall, "unreachable.");
  }
  return table_name;
}

inline int get_column_wise_sharding_factor(TableVariant v) {
  int column_wise_sharding_factor;
  if (std::get_if<std::string>(&v)) {
    column_wise_sharding_factor = 1;
  } else if (std::tuple<std::string, int> *tuple_ptr =
                 std::get_if<std::tuple<std::string, int>>(&v)) {
    column_wise_sharding_factor = std::get<1>(*tuple_ptr);
  } else {
    HCTR_OWN_THROW(Error_t::IllegalCall, "unreachable.");
  }
  return column_wise_sharding_factor;
}

class EmbeddingCollectionConfig {
 public:
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::vector<int> dr_lookup_ids_;

  using LookupConfig = std::pair<std::string, ::embedding::LookupParam>;
  std::vector<LookupConfig> lookup_configs_;

  std::vector<EmbeddingTableConfig> emb_table_config_list_;

  std::vector<ShardStrategy> shard_strategy_;
  std::vector<std::vector<std::string>> shard_matrix_;
  using CompressionStrategyConfig =
      std::unordered_map<::embedding::CompressionStrategy, std::vector<std::string>>;
  CompressionStrategyConfig compression_strategy_config_;

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
             const std::vector<ShardStrategy> &shard_strategy,
             const CompressionStrategyConfig &compression_strategy_config) {
    shard_matrix_.clear();
    shard_strategy_.clear();
    compression_strategy_config_.clear();

    shard_matrix_ = shard_matrix;
    shard_strategy_ = shard_strategy;
    compression_strategy_config_ = compression_strategy_config;
    dr_lookup_ids_.resize(lookup_configs_.size());
    std::iota(dr_lookup_ids_.begin(), dr_lookup_ids_.end(), 0);
  }
};

EmbeddingCollectionConfig split_column_wise_sharding_config(
    const EmbeddingCollectionConfig &user_ebc_config);

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
                      "embedding_load can't find table name : ", name, " in model\n");
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
                    "create_lookup_params_from_ebc_config error, no such name: ", name, "\n");
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
                    "create_table_params_from_ebc_config error, no such name: ", name, "\n");
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
                      "create_shard_matrix_from_ebc_config error, no such name: ", name, "\n");
      HCTR_CHECK_HINT(table_name_to_id_dict.at(name) < num_table,
                      "create_shard_matrix_from_ebc_config error, name is out of range: ", name,
                      "\n");
      shard_matrix[gpu_id][table_name_to_id_dict.at(name)] = 1;
    }
  }
  return shard_matrix;
}

inline std::vector<::embedding::GroupedTableParam> create_grouped_embedding_param_from_ebc_config(
    const TableNameToIDDict &table_name_to_id_dict, const EmbeddingCollectionConfig &config) {
  std::vector<::embedding::GroupedTableParam> grouped_embedding_params;
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
    for (auto &table_tuple : group_strategy) {
      std::string table_name = get_table_name(table_tuple);
      int column_wise_sharding_factor = get_column_wise_sharding_factor(table_tuple);
      HCTR_CHECK_HINT(column_wise_sharding_factor == 1, "column-wise sharding factor must be 1");

      HCTR_CHECK_HINT(
          table_name_to_id_dict.find(table_name) != table_name_to_id_dict.end(),
          "create_grouped_embedding_param_from_ebc_config error, no such name: ", table_name, "\n");
      table_ids.push_back(table_name_to_id_dict.at(table_name));
    }
    // require ordered
    std::sort(table_ids.begin(), table_ids.end());
    ::embedding::GroupedTableParam grouped_emb_param{placement_strategy, table_ids};
    grouped_embedding_params.push_back(std::move(grouped_emb_param));
  }
  return grouped_embedding_params;
}

inline ::embedding::CompressionParam create_compression_param_from_ebc_config(
    const TableNameToIDDict &table_name_to_id_dict, const EmbeddingCollectionConfig &config) {
  ::embedding::CompressionParam compression_param;
  for (auto &[strategy, table_names] : config.compression_strategy_config_) {
    for (auto table_name : table_names) {
      HCTR_CHECK_HINT(
          table_name_to_id_dict.find(table_name) != table_name_to_id_dict.end(),
          "create_grouped_embedding_param_from_ebc_config error, no such name: ", table_name, "\n");
      compression_param.compression_strategy_to_table_ids[strategy].insert(
          table_name_to_id_dict.at(table_name));
    }
  }
  return compression_param;
}

}  // namespace HugeCTR

namespace embedding {

class EmbeddingCollection {
 private:
  std::shared_ptr<HugeCTR::ResourceManager> resource_manager_;

  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> embeddings_, eval_embeddings_;

  std::vector<std::vector<EmbeddingOutputAttr>> embedding_output_attrs_;
  std::vector<std::vector<Wgrad>> wgrad_list_;
  std::unique_ptr<HugeCTR::GPUBarrier> gpu_barrier_;

  void init_embedding_output_attrs(std::vector<std::shared_ptr<CoreResourceManager>> core);

  void init_wgrad(std::vector<std::shared_ptr<CoreResourceManager>> core,
                  std::shared_ptr<HugeCTR::ExchangeWgrad> exchange_wgrad);

  void init_peer_buffer(std::vector<std::shared_ptr<CoreResourceManager>> core);

  IGroupedEmbeddingTable *get_table(int gpu_id, size_t grouped_id) {
    int grouped_table_id = ebc_param_.grouped_lookup_params[grouped_id].grouped_table_idx;

    return embedding_tables_[gpu_id][grouped_table_id].get();
  }

 public:
  // Fix:load and dump use these , put it on public temporary
  std::vector<HugeCTR::OptParams> embedding_optimizers_;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingTable>>> embedding_tables_;
  EmbeddingCollectionParam ebc_param_;
  EmbeddingCollectionParam eval_ebc_param_;
  std::vector<EmbeddingTableParam> emb_table_param_list_;

 public:
  EmbeddingCollection(std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
                      std::vector<std::shared_ptr<CoreResourceManager>> core,
                      const EmbeddingCollectionParam &ebc_param,
                      const EmbeddingCollectionParam &eval_ebc_param,
                      const std::vector<EmbeddingTableParam> &emb_table_param_list,
                      std::shared_ptr<HugeCTR::ExchangeWgrad> exchange_wgrad = nullptr);

  void cache_ddl_output(int gpu_id, const HugeCTR::DataDistributor::Result &input,
                        HugeCTR::DataDistributor::Result &output, int batch_size);

  void forward_per_gpu(Stage stage, bool is_train, int gpu_id,
                       const HugeCTR::DataDistributor::Result &input, core23::Tensor &output_buffer,
                       int batch_size);

  void forward_per_gpu(bool is_train, int gpu_id, const HugeCTR::DataDistributor::Result &input,
                       core23::Tensor &output_buffer, int batch_size);

  void backward_per_gpu(Stage stage, int gpu_id, const HugeCTR::DataDistributor::Result &input,
                        const core23::Tensor &top_grad, int batch_size);

  void backward_per_gpu(int gpu_id, const HugeCTR::DataDistributor::Result &input,
                        const core23::Tensor &top_grad, int batch_size);

  void update_per_gpu(int gpu_id, EmbeddingGroupType embedding_group_type);

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

  const std::vector<Wgrad> &get_wgrad(int gpu_id) const { return wgrad_list_[gpu_id]; }
};

}  // namespace embedding
