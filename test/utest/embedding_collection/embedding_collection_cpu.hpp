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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iterator>
#include <numeric>
#include <unordered_set>

#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/embedding/embedding_planner.hpp"
#include "data_parallel_embedding_cpu.hpp"
#include "model_parallel_embedding_cpu.hpp"
#include "reference_embedding.hpp"
using namespace embedding;

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class EmbeddingCollectionCPU {
 public:
  int num_gpus_;
  int num_embedding_;
  int num_table_;
  EmbeddingCollectionParam user_ebc_param_;
  EmbeddingCollectionParam ebc_param_;
  std::vector<std::vector<EmbeddingShardingParam>> global_sharding_param_list_;
  std::vector<int> ev_offset_list_;
  std::vector<int> ev_size_list_;

  std::vector<int> unique_id_space_list_;
  std::vector<std::unordered_map<key_t, std::vector<float>>> grad_info_;

  std::vector<std::vector<emb_t>> embedding_vec_;

  std::vector<ModelParallelEmbeddingCPU<key_t, offset_t, index_t, emb_t>> mp_embedding_list_;
  std::vector<DataParallelEmbeddingCPU<key_t, offset_t, index_t, emb_t>> dp_embedding_list_;

  EmbeddingTableCPU<key_t, index_t> emb_table_cpu_;

  EmbeddingCollectionCPU(
      int num_gpus, int num_table, const EmbeddingCollectionParam &ebc_param,
      const std::vector<std::vector<EmbeddingShardingParam>> &global_sharding_param_list,
      std::vector<std::vector<IEmbeddingTable *>> emb_table_list,
      const std::vector<EmbeddingTableParam> &table_param_list)
      : num_gpus_(num_gpus),
        num_embedding_(-1),
        num_table_(num_table),
        user_ebc_param_(ebc_param),
        ebc_param_(ebc_param),
        global_sharding_param_list_(global_sharding_param_list),
        ev_offset_list_{0},
        emb_table_cpu_(num_table, emb_table_list, table_param_list) {
    flatten_concat_embedding(&ebc_param_, &global_sharding_param_list_);
    num_embedding_ = ebc_param_.num_embedding;
    for (int i = 0; i < num_embedding_; ++i) {
      ev_size_list_.push_back(ebc_param_.embedding_params[i].ev_size);
      ev_offset_list_.push_back(ebc_param_.embedding_params[i].ev_size);
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());

    for (auto emb_param : ebc_param_.embedding_params) {
      if (unique_id_space_list_.size() == 0 || emb_param.id_space > unique_id_space_list_.back()) {
        unique_id_space_list_.push_back(emb_param.id_space);
      }
    }
    grad_info_.resize(unique_id_space_list_.size());

    embedding_vec_.resize(num_gpus_);

    assert(global_sharding_param_list_.size() == num_gpus);
    assert(num_gpus > 0);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      assert(global_sharding_param_list_[0].size() == global_sharding_param_list_[gpu_id].size());
    }

    for (size_t idx_sharding_param = 0; idx_sharding_param < global_sharding_param_list_[0].size();
         ++idx_sharding_param) {
      std::vector<EmbeddingShardingParam> one_embedding_sharding_param;
      for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        one_embedding_sharding_param.push_back(
            global_sharding_param_list_[gpu_id][idx_sharding_param]);
      }
      if (one_embedding_sharding_param[0].table_placement_strategy ==
          TablePlacementStrategy::DataParallel) {
        dp_embedding_list_.emplace_back(num_gpus_, num_table_, ebc_param_,
                                        one_embedding_sharding_param);
      } else if (one_embedding_sharding_param[0].table_placement_strategy ==
                 TablePlacementStrategy::Localized) {
        mp_embedding_list_.emplace_back(num_gpus_, num_table_, ebc_param_,
                                        one_embedding_sharding_param);
      }
    }
  }

  std::vector<offset_t> flatten_bucket_range(const std::vector<offset_t> &bucket_range,
                                             int batch_size) {
    std::vector<offset_t> flatten_concat_bucket_range{0};
    for (int embedding_id = 0; embedding_id < user_ebc_param_.num_embedding; ++embedding_id) {
      for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int idx = batch_size * embedding_id + batch_id;
        if (user_ebc_param_.embedding_params[embedding_id].combiner == Combiner::Concat) {
          for (int i = 0; i < user_ebc_param_.embedding_params[embedding_id].hotness; ++i) {
            flatten_concat_bucket_range.push_back(1);
          }
        } else {
          flatten_concat_bucket_range.push_back(bucket_range[idx + 1] - bucket_range[idx]);
        }
      }
    }
    std::partial_sum(flatten_concat_bucket_range.begin(), flatten_concat_bucket_range.end(),
                     flatten_concat_bucket_range.begin());
    return flatten_concat_bucket_range;
  }

  std::vector<key_t> transpose_keys(const std::vector<key_t> &keys,
                                    const std::vector<offset_t> &bucket_range,
                                    const std::vector<offset_t> &t_bucket_range, int batch_size) {
    std::vector<int> embedding_offset;
    int num_embedding = 0;
    for (int embedding_id = 0; embedding_id < user_ebc_param_.num_embedding; ++embedding_id) {
      embedding_offset.push_back(num_embedding);
      num_embedding += (user_ebc_param_.embedding_params[embedding_id].combiner == Combiner::Concat)
                           ? user_ebc_param_.embedding_params[embedding_id].hotness
                           : 1;
    }
    embedding_offset.push_back(num_embedding);

    std::vector<key_t> t_keys;
    t_keys.resize(bucket_range.back());
    for (int embedding_id = 0; embedding_id < user_ebc_param_.num_embedding; ++embedding_id) {
      for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int idx = batch_size * embedding_id + batch_id;
        offset_t start = bucket_range[idx];
        offset_t end = bucket_range[idx + 1];

        if (user_ebc_param_.embedding_params[embedding_id].combiner == Combiner::Concat) {
          for (int i = 0; i < user_ebc_param_.embedding_params[embedding_id].hotness; ++i) {
            int flatten_embedding_id = embedding_offset[embedding_id] + i;
            int t_idx = flatten_embedding_id * batch_size + batch_id;
            offset_t t_start = t_bucket_range[t_idx];
            t_keys[t_start] = keys[start + i];
          }
        } else {
          int flatten_embedding_id = embedding_offset[embedding_id];
          int t_idx = flatten_embedding_id * batch_size + batch_id;
          offset_t t_start = t_bucket_range[t_idx];

          for (uint32_t i = 0; i < (end - start); ++i) {
            t_keys[t_start + i] = keys[i + start];
          }
        }
      }
    }
    // std::cout << "cpu transpose key:\n";
    // for (key_t k: t_keys) {
    //   std::cout << k << " ";
    // }
    // std::cout << "\n";
    return t_keys;
  }

  void transpose_forward_output(int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    std::vector<int> original_embedding_id_list;
    std::vector<int> start_embedding_id_list;
    std::vector<int> hotness_list;
    int num_flatten_embedding_ = 0;
    for (int embedding_id = 0; embedding_id < user_ebc_param_.num_embedding; ++embedding_id) {
      auto &emb_param = user_ebc_param_.embedding_params[embedding_id];
      hotness_list.push_back(emb_param.hotness);

      if (emb_param.combiner == Combiner::Concat) {
        for (int i = 0; i < emb_param.hotness; ++i) {
          start_embedding_id_list.push_back(num_flatten_embedding_);
          original_embedding_id_list.push_back(embedding_id);
        }
        num_flatten_embedding_ += emb_param.hotness;
      } else {
        start_embedding_id_list.push_back(num_flatten_embedding_);
        original_embedding_id_list.push_back(embedding_id);
        num_flatten_embedding_ += 1;
      }
    }
    start_embedding_id_list.push_back(num_flatten_embedding_);

    std::vector<std::vector<emb_t>> t_embedding_vec;
    t_embedding_vec.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      t_embedding_vec[gpu_id].clear();
      t_embedding_vec[gpu_id].resize(batch_size_per_gpu * ev_offset_list_.back());
    }

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      for (int embedding_id = 0; embedding_id < ebc_param_.num_embedding; ++embedding_id) {
        int ev_size = ebc_param_.embedding_params[embedding_id].ev_size;
        for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
          int start_idx = batch_size_per_gpu * ev_offset_list_[embedding_id] + batch_id * ev_size;
          if (ebc_param_.embedding_params[embedding_id].combiner == Combiner::Concat) {
            int start_embedding_id = start_embedding_id_list[embedding_id];
            int hotness = hotness_list[original_embedding_id_list[embedding_id]];

            int column = embedding_id - start_embedding_id;
            int row = batch_id;
            int t_local_bucket_id = row * hotness + column;
            int t_start_idx = batch_size_per_gpu * ev_offset_list_[start_embedding_id] +
                              t_local_bucket_id * ev_size;
            for (int i = 0; i < ev_size; ++i) {
              t_embedding_vec[gpu_id][t_start_idx + i] = embedding_vec_[gpu_id][start_idx + i];
            }
          } else {
            for (int i = 0; i < ev_size; ++i) {
              t_embedding_vec[gpu_id][start_idx + i] = embedding_vec_[gpu_id][start_idx + i];
            }
          }
        }
      }
    }

    embedding_vec_ = t_embedding_vec;
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / user_ebc_param_.num_embedding;
    int batch_size_per_gpu = batch_size / num_gpus_;

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      embedding_vec_[gpu_id].clear();
      embedding_vec_[gpu_id].resize(batch_size_per_gpu * ev_offset_list_.back());
    }

    auto flatten_concat_bucket_range = flatten_bucket_range(bucket_range, batch_size);
    auto t_keys = transpose_keys(keys, bucket_range, flatten_concat_bucket_range, batch_size);

    for (auto &embedding : mp_embedding_list_) {
      embedding.embedding_forward_cpu(t_keys, flatten_concat_bucket_range, emb_table_cpu_,
                                      embedding_vec_, batch_size);
    }
    for (auto &embedding : dp_embedding_list_) {
      embedding.embedding_forward_cpu(t_keys, flatten_concat_bucket_range, emb_table_cpu_,
                                      embedding_vec_, batch_size);
    }
    transpose_forward_output(batch_size);
  }

  std::vector<std::vector<emb_t>> transpose_backward_top_grad(
      const std::vector<std::vector<emb_t>> &top_grad, int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    std::vector<int> original_embedding_id_list;
    std::vector<int> start_embedding_id_list;
    std::vector<int> hotness_list;
    int num_flatten_embedding_ = 0;
    for (int embedding_id = 0; embedding_id < user_ebc_param_.num_embedding; ++embedding_id) {
      auto &emb_param = user_ebc_param_.embedding_params[embedding_id];
      hotness_list.push_back(emb_param.hotness);

      if (emb_param.combiner == Combiner::Concat) {
        for (int i = 0; i < emb_param.hotness; ++i) {
          start_embedding_id_list.push_back(num_flatten_embedding_);
          original_embedding_id_list.push_back(embedding_id);
        }
        num_flatten_embedding_ += emb_param.hotness;
      } else {
        original_embedding_id_list.push_back(embedding_id);
        start_embedding_id_list.push_back(num_flatten_embedding_);
        num_flatten_embedding_ += 1;
      }
    }
    start_embedding_id_list.push_back(num_flatten_embedding_);

    std::vector<std::vector<emb_t>> t_top_grad;
    t_top_grad.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      t_top_grad[gpu_id].clear();
      t_top_grad[gpu_id].resize(batch_size_per_gpu * ev_offset_list_.back());
    }
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      for (int embedding_id = 0; embedding_id < ebc_param_.num_embedding; ++embedding_id) {
        int ev_size = ebc_param_.embedding_params[embedding_id].ev_size;
        for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
          int start_idx = batch_size_per_gpu * ev_offset_list_[embedding_id] + batch_id * ev_size;
          if (ebc_param_.embedding_params[embedding_id].combiner == Combiner::Concat) {
            int start_embedding_id = start_embedding_id_list[embedding_id];
            int hotness = hotness_list[original_embedding_id_list[embedding_id]];

            int local_bucket_id = embedding_id - start_embedding_id + batch_id * hotness +
                                  start_embedding_id * batch_size_per_gpu;
            int t_embedding_id = local_bucket_id / batch_size_per_gpu;
            int t_batch_id = local_bucket_id % batch_size_per_gpu;
            int t_start_idx =
                batch_size_per_gpu * ev_offset_list_[t_embedding_id] + t_batch_id * ev_size;

            for (int i = 0; i < ev_size; ++i) {
              t_top_grad[gpu_id][start_idx + i] = top_grad[gpu_id][t_start_idx + i];
            }
          } else {
            for (int i = 0; i < ev_size; ++i) {
              t_top_grad[gpu_id][start_idx + i] = top_grad[gpu_id][start_idx + i];
            }
          }
        }
      }
    }
    return t_top_grad;
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grads, int batch_size) {
    auto t_top_grads = transpose_backward_top_grad(top_grads, batch_size);
    grad_info_.clear();
    grad_info_.resize(unique_id_space_list_.size());

    for (auto &embedding : mp_embedding_list_) {
      embedding.embedding_backward_cpu(t_top_grads, grad_info_, batch_size);
    }
    for (auto &embedding : dp_embedding_list_) {
      embedding.embedding_backward_cpu(t_top_grads, grad_info_, batch_size);
    }
  }

  void embedding_update_cpu() { emb_table_cpu_.update(grad_info_); }
};
