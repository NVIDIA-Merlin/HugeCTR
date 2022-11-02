/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "embedding/embedding.hpp"
#include "embedding/operators/transpose_input.hpp"
#include "embedding_storage/embedding_table.hpp"

namespace HugeCTR {
class EmbeddingPlanner {
 public:
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::vector<embedding::LookupParam> lookup_params_;
  std::set<int> emb_table_id_set_;
  std::vector<embedding::EmbeddingTableParam> emb_table_list_;
  std::vector<std::vector<int>> shard_matrix_;
  std::vector<std::vector<int>> emb_table_group_strategy_;
  std::vector<embedding::TablePlacementStrategy> emb_table_placement_strategy_;

  void embedding_lookup(const embedding::EmbeddingTableParam &emb_table_config,
                        const std::string &bottom_name, const std::string &top_name,
                        const std::string &combiner_str) {
    embedding::Combiner combiner;
    if (combiner_str == "concat") {
      combiner = embedding::Combiner::Concat;
    } else if (combiner_str == "sum") {
      combiner = embedding::Combiner::Sum;
    } else if (combiner_str == "average") {
      combiner = embedding::Combiner::Average;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, combiner_str + " is not supported.");
    }

    lookup_params_.emplace_back(lookup_params_.size(), emb_table_config.table_id, combiner, -1,
                                emb_table_config.ev_size);
    bottom_names_.push_back(bottom_name);
    top_names_.push_back(top_name);

    if (emb_table_id_set_.find(emb_table_config.table_id) == emb_table_id_set_.end()) {
      emb_table_id_set_.insert(emb_table_config.table_id);
      emb_table_list_.push_back(emb_table_config);
    }
  }

  const EmbeddingPlanner &create_embedding_collection(
      const std::vector<std::vector<int>> &shard_matrix,
      const std::vector<std::vector<int>> &emb_table_group_strategy,
      const std::vector<std::string> &emb_table_placement_strategy) {
    shard_matrix_.clear();
    emb_table_group_strategy_.clear();
    emb_table_placement_strategy_.clear();

    shard_matrix_ = shard_matrix;
    emb_table_group_strategy_ = emb_table_group_strategy;
    for (const auto &tps_str : emb_table_placement_strategy) {
      embedding::TablePlacementStrategy tps;
      if (tps_str == "mp") {
        tps = embedding::TablePlacementStrategy::ModelParallel;
      } else if (tps_str == "dp") {
        tps = embedding::TablePlacementStrategy::DataParallel;
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "table placement strategy " + tps_str + " not supported");
      }
      emb_table_placement_strategy_.push_back(tps);
    }
    return *this;
  }
};
}  // namespace HugeCTR

namespace embedding {

class EmbeddingCollection {
  EmbeddingCollectionParam ebc_param_;
  EmbeddingCollectionParam eval_ebc_param_;

  std::vector<std::unique_ptr<PreprocessInput>> preprocess_inputs_;
  std::vector<std::unique_ptr<PreprocessInput>> eval_preprocess_inputs_;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> embeddings_;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingOp>>> eval_embeddings_;
  std::vector<std::vector<std::unique_ptr<IGroupedEmbeddingTable>>> embedding_tables_;

  std::vector<std::vector<core::Tensor>> unique_key_list_;
  std::vector<std::vector<size_t>> num_unique_key_list_;
  std::vector<std::vector<core::Tensor>> num_unique_key_per_table_offset_list_;
  std::vector<std::vector<size_t>> num_table_offset_list_;  // num_table_offset = num_table + 1
  std::vector<std::vector<core::Tensor>> wgrad_list_;
  std::vector<std::vector<core::Tensor>> wgrad_idx_offset_list_;
  std::vector<std::vector<core::Tensor>> table_id_list_list_;

 public:
  EmbeddingCollection(std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
                      std::vector<std::shared_ptr<CoreResourceManager>> core,
                      const EmbeddingCollectionParam &ebc_param,
                      const EmbeddingCollectionParam &eval_ebc_param,
                      const std::vector<EmbeddingTableParam> &emb_table_param_list);

  void forward_per_gpu(bool is_train, int gpu_id, const Tensor &key, const Tensor &bucket_range,
                       size_t num_keys, Tensor &output_buffer);

  void backward_per_gpu(int gpu_id, const Tensor &top_grad, bool allreduce);

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