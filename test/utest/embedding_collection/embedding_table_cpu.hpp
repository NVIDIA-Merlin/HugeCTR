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
#include "HugeCTR/embedding_storage/embedding_table.hpp"
using namespace embedding;

template <typename key_t, typename index_t>
class EmbeddingTableCPU {
 public:
  std::vector<std::unordered_map<key_t, std::vector<float>>> emb_table_list_;
  std::vector<EmbeddingTableParam> table_param_list_;

  EmbeddingTableCPU(int num_table,
                    std::vector<std::vector<IGroupedEmbeddingTable *>> emb_table_list,
                    const std::vector<EmbeddingTableParam> &table_param_list)
      : table_param_list_(table_param_list) {
    emb_table_list_.resize(num_table);

    for (int table_id = 0; table_id < static_cast<int>(emb_table_list.size()); ++table_id) {
      for (int gpu_id = 0; gpu_id < static_cast<int>(emb_table_list[table_id].size()); ++gpu_id) {
        core::Tensor keys;
        core::Tensor id_space_offset;
        core::Tensor emb_table;
        core::Tensor emb_table_ev_size_list;
        core::Tensor emb_table_id_space_list;
        emb_table_list[table_id][gpu_id]->dump(&keys, &id_space_offset, &emb_table,
                                               &emb_table_ev_size_list, &emb_table_id_space_list);

        std::vector<key_t> cpu_keys;
        keys.to(&cpu_keys);

        std::vector<index_t> cpu_id_space_offset;
        id_space_offset.to(&cpu_id_space_offset);

        std::vector<float> cpu_emb_table;
        emb_table.to(&cpu_emb_table);

        std::vector<int> cpu_emb_table_ev_size_list;
        emb_table_ev_size_list.to(&cpu_emb_table_ev_size_list);

        std::vector<int> cpu_emb_table_id_space_list;
        emb_table_id_space_list.to(&cpu_emb_table_id_space_list);

        std::vector<uint64_t> cpu_emb_table_ev_offset_list{0};
        for (int idx = 0; idx < static_cast<int>(cpu_emb_table_id_space_list.size()); ++idx) {
          index_t start = cpu_id_space_offset[idx];
          index_t end = cpu_id_space_offset[idx + 1];
          cpu_emb_table_ev_offset_list.push_back((end - start) * cpu_emb_table_ev_size_list[idx]);
        }
        std::partial_sum(cpu_emb_table_ev_offset_list.begin(), cpu_emb_table_ev_offset_list.end(),
                         cpu_emb_table_ev_offset_list.begin());

        for (int idx = 0; idx < static_cast<int>(cpu_emb_table_id_space_list.size()); ++idx) {
          int id_space = cpu_emb_table_id_space_list[idx];
          int ev_size = cpu_emb_table_ev_size_list[idx];
          uint64_t ev_offset = cpu_emb_table_ev_offset_list[idx];
          index_t start = cpu_id_space_offset[idx];
          index_t end = cpu_id_space_offset[idx + 1];
          for (index_t r = 0; r < (end - start); ++r) {
            key_t k = cpu_keys[start + r];
            std::vector<float> ev;
            for (int e = 0; e < ev_size; ++e) {
              ev.push_back(cpu_emb_table[ev_offset + r * ev_size + e]);
            }
            emb_table_list_[id_space][k] = ev;
          }
        }
      }
    }
  }

  void update(const std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info) {
    ASSERT_EQ(grad_info.size(), table_param_list_.size());
    int num_id_space = grad_info.size();
    for (int id_space = 0; id_space < num_id_space; ++id_space) {
      float lr = table_param_list_[id_space].opt_param.lr;
      float scaler = table_param_list_[id_space].opt_param.scaler;

      if (table_param_list_[id_space].opt_param.optimizer == HugeCTR::Optimizer_t::SGD) {
        for (auto &[key, ev] : grad_info[id_space]) {
          ASSERT_TRUE(emb_table_list_[id_space].find(key) != emb_table_list_[id_space].end());
          ASSERT_EQ(emb_table_list_[id_space][key].size(), ev.size());

          for (size_t i = 0; i < ev.size(); ++i) {
            float gi = (-lr * ev[i] / scaler);
            emb_table_list_[id_space][key][i] += gi;
          }
        }
      }
    }
  }
};
