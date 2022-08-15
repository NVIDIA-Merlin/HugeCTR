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
#include "common.hpp"
namespace embedding {

std::ostream &operator<<(std::ostream &os, const Combiner &p) {
  switch (p) {
    case Combiner::Sum:
      os << "sum";
      break;
    case Combiner::Average:
      os << "average";
      break;
    case Combiner::Concat:
      os << "concat";
      break;
    default:
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const EmbeddingParam &p) {
  os << "embedding_id:" << p.embedding_id << ",";
  os << "id_space:" << p.id_space << ",";
  os << "combiner:" << p.combiner << ",";
  os << "hotness:" << p.hotness << ",";
  os << "ev_size:" << p.ev_size;
  return os;
}

EmbeddingShardingParam::EmbeddingShardingParam(int num_embedding,
                                               const EmbeddingShardParam &shard_param,
                                               int global_gpu_id)
    : table_placement_strategy(shard_param.table_placement_strategy) {
  shard_id = -1;
  for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
    if (shard_param.shard_matrix[global_gpu_id][embedding_id] < 0) {
      continue;
    }
    local_embedding_list.push_back(embedding_id);
    if (shard_id < 0) {
      shard_id = shard_param.shard_matrix[global_gpu_id][embedding_id];
      shards_count = shard_param.shard_count_list[embedding_id];
    } else {
      HCTR_CHECK_HINT(shard_id == shard_param.shard_matrix[global_gpu_id][embedding_id],
                      "Current implementation does not support multiple shard id in one gpu");
      HCTR_CHECK_HINT(shards_count == shard_param.shard_count_list[embedding_id],
                      "Current implementation does not support multiple num sharding in one gpu");
    }
  }

  int global_gpu_count = static_cast<int>(shard_param.shard_matrix.size());
  global_embedding_list.resize(global_gpu_count);
  for (int gpu_id = 0; gpu_id < global_gpu_count; ++gpu_id) {
    for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
      if (shard_param.shard_matrix[gpu_id][embedding_id] < 0) {
        continue;
      }
      global_embedding_list[gpu_id].push_back(embedding_id);
    }
  }
}

EmbeddingShardParam::EmbeddingShardParam(const std::vector<std::vector<int>> _shard_matrix,
                                         TablePlacementStrategy _tps)
    : shard_matrix(_shard_matrix), table_placement_strategy(_tps) {
  HCTR_CHECK_HINT(shard_matrix.size() > 0, "empty shard matrix");
  int num_embedding = static_cast<int>(shard_matrix[0].size());
  int num_gpu = static_cast<int>(shard_matrix.size());
  shard_count_list.assign(num_embedding, 0);
  for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
    for (int gpu_id = 0; gpu_id < num_gpu; ++gpu_id) {
      if (shard_matrix[gpu_id][embedding_id] < 0) continue;
      if (table_placement_strategy == TablePlacementStrategy::DataParallel) {
        HCTR_CHECK_HINT(shard_matrix[gpu_id][embedding_id] == num_gpu,
                        "data parallel shard matrix check error");
      }
      shard_count_list[embedding_id] += 1;
    }
  }
  if (table_placement_strategy == TablePlacementStrategy::DataParallel) {
    for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
      HCTR_CHECK_HINT(
          shard_count_list[embedding_id] == 0 || shard_count_list[embedding_id] == num_gpu,
          "data parallel shard matrix check error");
    }
  }
}

void flatten_concat_embedding(EmbeddingCollectionParam *_ebc_param,
                              std::vector<std::vector<EmbeddingShardingParam>> *_ebs_param) {
  auto &ebc_param = *_ebc_param;
  auto &ebs_param = *_ebs_param;

  for (size_t gpu_id = 0; gpu_id < ebs_param.size(); ++gpu_id) {
    for (size_t sharding_param_id = 0; sharding_param_id < ebs_param[gpu_id].size();
         ++sharding_param_id) {
      auto &local_embedding_list = ebs_param[gpu_id][sharding_param_id].local_embedding_list;
      if (ebs_param[gpu_id][sharding_param_id].table_placement_strategy ==
          TablePlacementStrategy::DataParallel) {
        std::vector<int> local_id_space;
        for (int embedding_id : local_embedding_list) {
          local_id_space.push_back(ebc_param.embedding_params[embedding_id].id_space);
        }
        size_t num_local_id_space = local_id_space.size();

        std::sort(local_id_space.begin(), local_id_space.end());
        auto last = std::unique(local_id_space.begin(), local_id_space.end());
        local_id_space.erase(last, local_id_space.end());
        HCTR_CHECK_HINT(
            local_id_space.size() == num_local_id_space,
            "Illegal input. DataParallel embedding should not have shared embedding table");
      }
    }
  }

  auto &embedding_params = ebc_param.embedding_params;
  std::vector<int> flatten_concat_embedding_offset;
  int num_embedding_ = 0;
  for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
    flatten_concat_embedding_offset.push_back(num_embedding_);
    num_embedding_ += (embedding_params[embedding_id].combiner == Combiner::Concat)
                          ? embedding_params[embedding_id].hotness
                          : 1;
  }
  flatten_concat_embedding_offset.push_back(num_embedding_);
  if (num_embedding_ == ebc_param.num_embedding) return;

  for (size_t gpu_id = 0; gpu_id < ebs_param.size(); ++gpu_id) {
    for (size_t sharding_param_id = 0; sharding_param_id < ebs_param[gpu_id].size();
         ++sharding_param_id) {
      std::vector<int> flatten_local_embedding_list;
      auto &local_embedding_list = ebs_param[gpu_id][sharding_param_id].local_embedding_list;
      for (int embedding_id : local_embedding_list) {
        if (embedding_params[embedding_id].combiner == Combiner::Concat) {
          HCTR_CHECK_HINT(ebs_param[gpu_id][sharding_param_id].table_placement_strategy !=
                              TablePlacementStrategy::DataParallel,
                          "DataParallel Embedding does not support concat combiner");
          for (int i = 0; i < embedding_params[embedding_id].hotness; ++i) {
            flatten_local_embedding_list.push_back(flatten_concat_embedding_offset[embedding_id] +
                                                   i);
          }
        } else {
          flatten_local_embedding_list.push_back(flatten_concat_embedding_offset[embedding_id]);
        }
      }
      ebs_param[gpu_id][sharding_param_id].local_embedding_list = flatten_local_embedding_list;

      std::vector<std::vector<int>> flatten_global_embedding_list;
      auto &global_embedding_list = ebs_param[gpu_id][sharding_param_id].global_embedding_list;
      flatten_global_embedding_list.resize(global_embedding_list.size());
      for (size_t global_gpu_id = 0; global_gpu_id < global_embedding_list.size();
           ++global_gpu_id) {
        for (int local_embedding_id : global_embedding_list[global_gpu_id]) {
          if (embedding_params[local_embedding_id].combiner == Combiner::Concat) {
            for (int i = 0; i < embedding_params[local_embedding_id].hotness; ++i) {
              flatten_global_embedding_list[global_gpu_id].push_back(
                  flatten_concat_embedding_offset[local_embedding_id] + i);
            }
          } else {
            flatten_global_embedding_list[global_gpu_id].push_back(
                flatten_concat_embedding_offset[local_embedding_id]);
          }
        }
      }
      ebs_param[gpu_id][sharding_param_id].global_embedding_list = flatten_global_embedding_list;
    }
  }

  std::vector<EmbeddingParam> embedding_params_;
  for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
    auto cur_emb_param = embedding_params[embedding_id];
    if (cur_emb_param.combiner == Combiner::Concat) {
      for (int i = 0; i < cur_emb_param.hotness; ++i) {
        EmbeddingParam seperate_concat_emb_param;
        seperate_concat_emb_param.embedding_id = flatten_concat_embedding_offset[embedding_id] + i;
        seperate_concat_emb_param.id_space = cur_emb_param.id_space;
        seperate_concat_emb_param.combiner = cur_emb_param.combiner;
        seperate_concat_emb_param.hotness = 1;
        seperate_concat_emb_param.ev_size = cur_emb_param.ev_size;
        embedding_params_.push_back(std::move(seperate_concat_emb_param));
      }
    } else {
      cur_emb_param.embedding_id = flatten_concat_embedding_offset[embedding_id];
      embedding_params_.push_back(cur_emb_param);
    }
  }
  ebc_param.num_embedding = num_embedding_;
  ebc_param.embedding_params = embedding_params_;
}
}  // namespace embedding
