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

class EmbeddingTablePlaceholder {
 public:
  embedding::EmbeddingTableParam param_;

  EmbeddingTablePlaceholder(int table_id, int max_vocabulary_size, int ev_size, int64_t min_key,
                            int64_t max_key, std::optional<OptParams> opt_params) {
    param_.id_space = table_id; // TODO: make them consistent
    param_.max_vocabulary_size = max_vocabulary_size;
    param_.ev_size = ev_size;
    param_.min_key = min_key;
    param_.max_key = max_key;

    if (opt_params.has_value()) {
      param_.opt_param = opt_params.value();
    } else {
      param_.opt_param.optimizer = Optimizer_t::NOT_INITIALIZED;
    }
  }
};

class EmbeddingCollectionPlaceholder {
 public:
  std::string plan_file_;
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::vector<EmbeddingTablePlaceholder> emb_table_placeholder_;

  EmbeddingCollectionPlaceholder(
      const std::string &plan_file, const std::vector<embedding::EmbeddingParam> &param,
      const std::vector<std::string> &bottom_names, const std::vector<std::string> &top_names,
      const std::vector<EmbeddingTablePlaceholder> &emb_table_placeholder)
      : plan_file_(plan_file),
        param_(param),
        bottom_names_(bottom_names),
        top_names_(top_names),
        emb_table_placeholder_(emb_table_placeholder) {}
};

class EmbeddingPlanner {
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> bottom_names_;
  std::vector<std::string> top_names_;
  std::set<int> emb_table_id_set_;
  std::vector<EmbeddingTablePlaceholder> emb_table_placeholder_;

 public:
  EmbeddingPlanner() {}

  void embedding_lookup(const EmbeddingTablePlaceholder &emb_table, const std::string &bottom_name,
                        const std::string &top_name, const std::string &combiner) {
    embedding::EmbeddingParam emb_param;
    emb_param.embedding_id = param_.size();
    emb_param.id_space = emb_table.param_.id_space; // TODO: change to table_id
    if (combiner == "concat") {
      emb_param.combiner = embedding::Combiner::Concat;
    } else if (combiner == "sum") {
      emb_param.combiner = embedding::Combiner::Sum;
    } else if (combiner == "average") {
      emb_param.combiner = embedding::Combiner::Average;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, combiner + " is not supported.");
    }
    emb_param.ev_size = emb_table.param_.ev_size;
    emb_param.hotness = -1;  // placeholder
    param_.push_back(std::move(emb_param));
    bottom_names_.push_back(bottom_name);
    top_names_.push_back(top_name);
    if (emb_table_id_set_.find(emb_table.param_.id_space) == emb_table_id_set_.end()) {
      emb_table_id_set_.insert(emb_table.param_.id_space);
      emb_table_placeholder_.push_back(emb_table);
    }
  }

  EmbeddingCollectionPlaceholder create_embedding_collection(const std::string &plan_file) {
    std::sort(emb_table_placeholder_.begin(), emb_table_placeholder_.end(),
              [](const EmbeddingTablePlaceholder &lhs, const EmbeddingTablePlaceholder &rhs) {
                return lhs.param_.table_id < rhs.param_.table_id;
              });
    return EmbeddingCollectionPlaceholder(plan_file, param_, bottom_names_, top_names_,
                                          emb_table_placeholder_);
  }
};

}  // namespace HugeCTR