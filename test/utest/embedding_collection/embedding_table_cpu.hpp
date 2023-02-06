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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <embedding/embedding.hpp>
#include <embedding_storage/embedding_table.hpp>
#include <iterator>
#include <numeric>
#include <unordered_set>

using namespace embedding;

template <typename key_t, typename index_t>
class EmbeddingTableCPU {
 public:
  std::vector<std::unordered_map<key_t, std::vector<float>>> emb_table_list_;
  std::vector<EmbeddingTableParam> table_param_list_;

  EmbeddingTableCPU(int num_table, std::vector<std::vector<IGroupedEmbeddingTable *>> emb_storages,
                    const std::vector<EmbeddingTableParam> &table_param_list)
      : table_param_list_(table_param_list) {
    emb_table_list_.resize(num_table);

    for (int gpu_id = 0; gpu_id < static_cast<int>(emb_storages.size()); ++gpu_id) {
      for (int storage_id = 0; storage_id < static_cast<int>(emb_storages[gpu_id].size());
           ++storage_id) {
        core::Tensor keys;
        core::Tensor num_key_per_table_offset;
        core::Tensor emb_table;
        core::Tensor emb_table_ev_size;
        core::Tensor emb_table_ids;
        emb_storages[gpu_id][storage_id]->dump(&keys, &num_key_per_table_offset, &emb_table,
                                               &emb_table_ev_size, &emb_table_ids);

        std::vector<key_t> cpu_keys;
        keys.to(&cpu_keys);

        std::vector<index_t> cpu_num_key_per_table_offset;
        num_key_per_table_offset.to(&cpu_num_key_per_table_offset);

        std::vector<float> cpu_emb_table;
        emb_table.to(&cpu_emb_table);

        std::vector<int> cpu_emb_table_ev_size;
        emb_table_ev_size.to(&cpu_emb_table_ev_size);

        std::vector<int> cpu_emb_table_ids;
        emb_table_ids.to(&cpu_emb_table_ids);

        int ev_start = 0;
        for (size_t i = 0; i < cpu_emb_table_ids.size(); ++i) {
          int table_id = cpu_emb_table_ids[i];
          int ev_size = cpu_emb_table_ev_size[i];
          HCTR_CHECK_HINT(table_id < static_cast<int>(table_param_list_.size()),
                          "table id out of range in EmbeddingTableCPU");
          HCTR_CHECK_HINT(table_param_list_[table_id].ev_size == ev_size,
                          "ev size not match in EmbeddingTableCPU");
          index_t start = cpu_num_key_per_table_offset[i];
          index_t end = cpu_num_key_per_table_offset[i + 1];
          for (index_t r = 0; r < (end - start); ++r) {
            HCTR_CHECK_HINT(start + r < cpu_keys.size(),
                            "(start + r) out of range in EmbeddingTableCPU");
            key_t k = cpu_keys[start + r];
            std::vector<float> ev;
            for (int e = 0; e < ev_size; ++e) {
              HCTR_CHECK_HINT(ev_start + e < cpu_emb_table.size(),
                              "(ev_start + e) out of range in EmbeddingTableCPU");
              ev.push_back(cpu_emb_table[ev_start + e]);
            }
            if (emb_table_list_[table_id].find(k) != emb_table_list_[table_id].end()) {
              HCTR_CHECK_HINT(ev.size() == emb_table_list_[table_id][k].size(),
                              "ev not match in EmbeddingTableCPU");
              for (int e = 0; e < ev_size; ++e) {
                HCTR_CHECK_HINT(std::abs(ev[e] - emb_table_list_[table_id][k][e]) < 1e-3,
                                "ev not match in EmbeddingTableCPU");
              }
            } else {
              emb_table_list_[table_id][k] = ev;
            }
            ev_start += ev_size;
          }
        }
      }
    }
  }

  void update(const std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info) {
    ASSERT_EQ(grad_info.size(), table_param_list_.size());
    int num_table = grad_info.size();
    for (int table_id = 0; table_id < num_table; ++table_id) {
      float lr = table_param_list_[table_id].opt_param.lr;
      float scaler = table_param_list_[table_id].opt_param.scaler;

      if (table_param_list_[table_id].opt_param.optimizer == HugeCTR::Optimizer_t::SGD) {
        for (auto &[key, ev] : grad_info[table_id]) {
          ASSERT_TRUE(emb_table_list_[table_id].find(key) != emb_table_list_[table_id].end());
          ASSERT_EQ(emb_table_list_[table_id][key].size(), ev.size());

          for (size_t i = 0; i < ev.size(); ++i) {
            float gi = (-lr * ev[i] / scaler);
            emb_table_list_[table_id][key][i] += gi;
          }
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "EmbeddingTableCPU not support optimizer type.");
      }
    }
  }
};
