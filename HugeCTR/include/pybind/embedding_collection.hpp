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
#include <optional>

#include "HugeCTR/embedding/common.hpp"
#include "HugeCTR/embedding_storage/common.hpp"
#include "HugeCTR/include/optimizer.hpp"

namespace HugeCTR {

class EmbeddingTableConfig {
 public:
  embedding::EmbeddingTableParam param_;

  EmbeddingTableConfig(int table_id, int max_vocabulary_size, int ev_size, int64_t min_key,
                       int64_t max_key, std::optional<OptParams> opt_params) {
    param_.table_id = table_id;  // TODO: make them consistent
    param_.max_vocabulary_size = max_vocabulary_size;
    param_.ev_size = ev_size;
    param_.min_key = min_key;
    param_.max_key = max_key;

    if (opt_params.has_value()) {
      param_.opt_param = opt_params.value();
    } else {
      param_.opt_param.optimizer = Optimizer_t::NOT_INITIALIZED;
    }
    param_.init_param.initializer_type = HugeCTR::Initializer_t::Default;
  }
};

class EmbeddingCollectionPlaceholder {
 public:
  std::string plan_file_;
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::vector<EmbeddingTableConfig> emb_table_config_list_;

  EmbeddingCollectionPlaceholder(const std::string &plan_file,
                                 const std::vector<embedding::EmbeddingParam> &param,
                                 const std::vector<std::string> &bottom_names,
                                 const std::vector<std::string> &top_names,
                                 const std::vector<EmbeddingTableConfig> &emb_table_config_list)
      : plan_file_(plan_file),
        param_(param),
        bottom_names_(bottom_names),
        top_names_(top_names),
        emb_table_config_list_(emb_table_config_list) {}
};

class EmbeddingPlanner {
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::set<int> emb_table_id_set_;
  std::vector<EmbeddingTableConfig> emb_table_config_list_;

 public:
  EmbeddingPlanner() {}

  void embedding_lookup(const EmbeddingTableConfig &emb_table_config,
                        const std::string &bottom_name, const std::string &top_name,
                        const std::string &combiner) {
    embedding::EmbeddingParam emb_param;
    emb_param.embedding_id = param_.size();
    emb_param.id_space = emb_table_config.param_.table_id;  // TODO: change to table_id
    if (combiner == "concat") {
      emb_param.combiner = embedding::Combiner::Concat;
    } else if (combiner == "sum") {
      emb_param.combiner = embedding::Combiner::Sum;
    } else if (combiner == "average") {
      emb_param.combiner = embedding::Combiner::Average;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, combiner + " is not supported.");
    }
    emb_param.ev_size = emb_table_config.param_.ev_size;
    emb_param.hotness = -1;  // placeholder
    param_.push_back(std::move(emb_param));
    bottom_names_.push_back(bottom_name);
    top_names_.push_back(top_name);
    if (emb_table_id_set_.find(emb_table_config.param_.table_id) == emb_table_id_set_.end()) {
      emb_table_id_set_.insert(emb_table_config.param_.table_id);
      emb_table_config_list_.push_back(emb_table_config);
    }
  }

  EmbeddingCollectionPlaceholder create_embedding_collection(const std::string &plan_file) {
    std::sort(emb_table_config_list_.begin(), emb_table_config_list_.end(),
              [](const EmbeddingTableConfig &lhs, const EmbeddingTableConfig &rhs) {
                return lhs.param_.table_id < rhs.param_.table_id;
              });
    return EmbeddingCollectionPlaceholder(plan_file, param_, bottom_names_, top_names_,
                                          emb_table_config_list_);
  }
};

}  // namespace HugeCTR