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
using namespace embedding;

namespace meta {

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class EmbeddingReferenceCPU {
 public:
  int num_gpus_;
  int num_lookup_;
  std::vector<LookupParam> lookup_params_;
  int num_table_;
  EmbeddingTableCPU<key_t, index_t> emb_table_cpu_;

  std::vector<int> ev_offset_list_;

  std::vector<std::vector<emb_t>> embedding_vec_;
  std::vector<std::unordered_map<key_t, std::vector<float>>> accumulate_grad_map_;

  EmbeddingReferenceCPU(int num_gpus, const EmbeddingCollectionParam &ebc_param, int num_table,
                        const std::vector<EmbeddingTableParam> &table_param_list,
                        std::vector<std::vector<IGroupedEmbeddingTable *>> emb_storages)
      : num_gpus_(num_gpus),
        num_lookup_(ebc_param.num_lookup),
        lookup_params_(ebc_param.lookup_params),
        num_table_(num_table),
        emb_table_cpu_{num_table, emb_storages, table_param_list},
        ev_offset_list_{0} {
    HCTR_CHECK_HINT(num_gpus == static_cast<int>(emb_storages.size()),
                    "EmbeddingReferenceCPU emb_storages size not match with num_gpus");
    for (int i = 0; i < num_lookup_; ++i) {
      if (ebc_param.lookup_params[i].combiner == Combiner::Concat) {
        ev_offset_list_.push_back(ebc_param.lookup_params[i].max_hotness *
                                  ebc_param.lookup_params[i].ev_size);
      } else {
        ev_offset_list_.push_back(ebc_param.lookup_params[i].ev_size);
      }
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_lookup_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    embedding_vec_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      embedding_vec_[gpu_id].clear();
      embedding_vec_[gpu_id].resize(ev_offset_list_.back() * batch_size_per_gpu);
    }

    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      int table_id = lookup_params_[lookup_id].table_id;
      int ev_size = lookup_params_[lookup_id].ev_size;
      int ev_offset = ev_offset_list_[lookup_id];
      Combiner combiner = lookup_params_[lookup_id].combiner;
      int max_hotness = lookup_params_[lookup_id].max_hotness;

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = lookup_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        std::vector<std::vector<float>> lookup_evs;
        for (int r = 0; r < static_cast<int>(end - start); ++r) {
          key_t k = keys[start + r];
          ASSERT_TRUE(table_id < emb_table_cpu_.emb_table_list_.size());
          ASSERT_TRUE(emb_table_cpu_.emb_table_list_[table_id].find(k) !=
                      emb_table_cpu_.emb_table_list_[table_id].end());
          lookup_evs.push_back(emb_table_cpu_.emb_table_list_[table_id][k]);
        }
        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          for (int e = 0; e < ev_size; ++e) {
            float v = 0.f;

            for (int r = 0; r < static_cast<int>(end - start); ++r) {
              v += lookup_evs[r][e];
            }
            if (combiner == Combiner::Average && (end - start > 0)) {
              v /= (end - start);
            }
            int dst = ev_offset * batch_size_per_gpu + local_b * ev_size + e;
            embedding_vec_[gpu_id][dst] = HugeCTR::TypeConvert<emb_t, float>::convert(v);
          }
        } else {
          for (int r = 0; r < static_cast<int>(end - start); ++r) {
            for (int e = 0; e < ev_size; ++e) {
              int dst = ev_offset * batch_size_per_gpu + local_b * max_hotness * ev_size +
                        r * ev_size + e;
              embedding_vec_[gpu_id][dst] =
                  HugeCTR::TypeConvert<emb_t, float>::convert(lookup_evs[r][e]);
            }
          }
        }
      }
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grad,
                              const std::vector<key_t> &keys,
                              const std::vector<offset_t> &bucket_range) {
    accumulate_grad_map_.clear();
    accumulate_grad_map_.resize(num_table_);

    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_lookup_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      int table_id = lookup_params_[lookup_id].table_id;
      int ev_size = lookup_params_[lookup_id].ev_size;
      int ev_offset = ev_offset_list_[lookup_id];
      Combiner combiner = lookup_params_[lookup_id].combiner;
      int max_hotness = lookup_params_[lookup_id].max_hotness;

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = lookup_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        std::vector<float> grad_ev;

        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          for (int e = 0; e < ev_size; ++e) {
            float gi = HugeCTR::TypeConvert<float, emb_t>::convert(
                top_grad[gpu_id][ev_offset * batch_size_per_gpu + local_b * ev_size + e]);
            grad_ev.push_back(gi);
          }
        } else {
          for (int e = 0; e < max_hotness * ev_size; ++e) {
            float gi = HugeCTR::TypeConvert<float, emb_t>::convert(
                top_grad[gpu_id]
                        [ev_offset * batch_size_per_gpu + local_b * max_hotness * ev_size + e]);
            grad_ev.push_back(gi);
          }
        }

        if (combiner == Combiner::Average) {
          for (size_t i = 0; i < grad_ev.size(); ++i) {
            grad_ev[i] /= (end - start);
          }
        }
        for (int r = 0; r < static_cast<int>(end - start); ++r) {
          key_t k = keys[start + r];
          if (accumulate_grad_map_[table_id].find(k) == accumulate_grad_map_[table_id].end()) {
            accumulate_grad_map_[table_id][k].assign(ev_size, 0.f);
          }
          for (int e = 0; e < ev_size; ++e) {
            float gi;
            if (combiner == Combiner::Sum || combiner == Combiner::Average) {
              gi = grad_ev[e];
            } else {
              gi = grad_ev[r * ev_size + e];
            }
            accumulate_grad_map_[table_id][k][e] += gi;
          }
        }
      }
    }
  }

  void embedding_update_cpu() { emb_table_cpu_.update(accumulate_grad_map_); }
};
}  // namespace meta
