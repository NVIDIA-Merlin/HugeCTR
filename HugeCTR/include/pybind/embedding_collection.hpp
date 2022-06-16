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

class EmbeddingTablePlaceHolder {
 public:
  embedding::EmbeddingTableParam param_;

  EmbeddingTablePlaceHolder(int id_space, int max_vocabulary_size, int ev_size, int64_t min_key,
                            int64_t max_key, std::optional<OptParams> opt_params) {
    param_.id_space = id_space;
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

class EmbeddingCollectionPlaceHolder {
 public:
  std::string plan_file_;
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<EmbeddingTablePlaceHolder> emb_table_place_holder_;

  EmbeddingCollectionPlaceHolder(
      const std::string &plan_file, const std::vector<embedding::EmbeddingParam> &param,
      const std::vector<std::string> &input_names, const std::vector<std::string> &output_names,
      const std::vector<EmbeddingTablePlaceHolder> &emb_table_place_holder)
      : plan_file_(plan_file),
        param_(param),
        input_names_(input_names),
        output_names_(output_names),
        emb_table_place_holder_(emb_table_place_holder) {}
};

class EmbeddingPlanner {
  std::vector<embedding::EmbeddingParam> param_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::set<int> emb_table_id_space_set_;
  std::vector<EmbeddingTablePlaceHolder> emb_table_place_holder_;

 public:
  EmbeddingPlanner() {}

  void embedding_lookup(const EmbeddingTablePlaceHolder &emb_table, const std::string &bottom_name,
                        const std::string &top_name, const std::string &combiner) {
    embedding::EmbeddingParam emb_param;
    emb_param.embedding_id = param_.size();
    emb_param.id_space = emb_table.param_.id_space;
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
    emb_param.hotness = -1;  // place holder
    param_.push_back(std::move(emb_param));
    input_names_.push_back(bottom_name);
    output_names_.push_back(top_name);
    if (emb_table_id_space_set_.find(emb_table.param_.id_space) == emb_table_id_space_set_.end()) {
      emb_table_id_space_set_.insert(emb_table.param_.id_space);
      emb_table_place_holder_.push_back(emb_table);
    }
  }

  EmbeddingCollectionPlaceHolder create_embedding_collection(const std::string &plan_file) {
    std::sort(emb_table_place_holder_.begin(), emb_table_place_holder_.end(),
              [](const EmbeddingTablePlaceHolder &lhs, const EmbeddingTablePlaceHolder &rhs) {
                return lhs.param_.id_space < rhs.param_.id_space;
              });
    return EmbeddingCollectionPlaceHolder(plan_file, param_, input_names_, output_names_,
                                          emb_table_place_holder_);
  }
};

}  // namespace HugeCTR