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
using namespace embedding;

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class EmbeddingReferenceCPU {
 public:
  int num_gpus_;
  int num_embedding_;
  int num_table_;
  std::vector<int> id_space_list_;
  std::vector<int> ev_size_list_;
  std::vector<int> ev_offset_list_;
  std::vector<int> hotness_list_;
  std::vector<Combiner> combiner_list_;

  EmbeddingTableCPU<key_t, index_t> emb_table_cpu_;

  std::vector<std::vector<emb_t>> embedding_vec_;
  std::vector<std::unordered_map<key_t, std::vector<float>>> accumulate_grad_map_;

  EmbeddingReferenceCPU(int num_gpus, int num_table, const EmbeddingCollectionParam &ebc_param,
                        std::vector<std::vector<IEmbeddingTable *>> emb_table_list,
                        const std::vector<EmbeddingTableParam> &table_param_list)
      : num_gpus_(num_gpus),
        num_embedding_(ebc_param.num_embedding),
        num_table_(num_table),
        ev_offset_list_{0},
        emb_table_cpu_{num_table, emb_table_list, table_param_list} {
    for (int i = 0; i < num_embedding_; ++i) {
      id_space_list_.push_back(ebc_param.embedding_params[i].id_space);
      ev_size_list_.push_back(ebc_param.embedding_params[i].ev_size);
      hotness_list_.push_back(ebc_param.embedding_params[i].hotness);
      combiner_list_.push_back(ebc_param.embedding_params[i].combiner);
      if (ebc_param.embedding_params[i].combiner == Combiner::Concat) {
        ev_offset_list_.push_back(ebc_param.embedding_params[i].hotness *
                                  ebc_param.embedding_params[i].ev_size);
      } else {
        ev_offset_list_.push_back(ebc_param.embedding_params[i].ev_size);
      }
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_embedding_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    embedding_vec_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      embedding_vec_[gpu_id].clear();
      embedding_vec_[gpu_id].resize(ev_offset_list_.back() * batch_size_per_gpu);
    }

    for (int embedding_id = 0; embedding_id < num_embedding_; ++embedding_id) {
      int id_space = id_space_list_[embedding_id];
      int ev_size = ev_size_list_[embedding_id];
      int ev_offset = ev_offset_list_[embedding_id];
      Combiner combiner = combiner_list_[embedding_id];
      int hotness = hotness_list_[embedding_id];

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = embedding_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          for (int e = 0; e < ev_size; ++e) {
            if (start == end) {
              embedding_vec_[gpu_id][ev_offset * batch_size_per_gpu + local_b * ev_size + e] =
                  HugeCTR::TypeConvert<emb_t, float>::convert(0.f);
              continue;
            }
            float v = 0.f;

            for (int r = 0; r < static_cast<int>(end - start); ++r) {
              key_t k = keys[start + r];
              v += emb_table_cpu_.emb_table_list_[id_space][k][e];
            }
            if (combiner == Combiner::Average) {
              v /= (end - start);
            }
            embedding_vec_[gpu_id][ev_offset * batch_size_per_gpu + local_b * ev_size + e] =
                HugeCTR::TypeConvert<emb_t, float>::convert(v);
          }
        } else {
          HCTR_ASSERT((end - start) == hotness);
          for (int r = 0; r < static_cast<int>(end - start); ++r) {
            key_t k = keys[start + r];
            for (int e = 0; e < ev_size; ++e) {
              embedding_vec_[gpu_id][ev_offset * batch_size_per_gpu + local_b * hotness * ev_size +
                                     r * ev_size + e] =
                  HugeCTR::TypeConvert<emb_t, float>::convert(
                      emb_table_cpu_.emb_table_list_[id_space][k][e]);
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

    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_embedding_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    for (int embedding_id = 0; embedding_id < num_embedding_; ++embedding_id) {
      int id_space = id_space_list_[embedding_id];
      int ev_size = ev_size_list_[embedding_id];
      int ev_offset = ev_offset_list_[embedding_id];
      Combiner combiner = combiner_list_[embedding_id];
      int hotness = hotness_list_[embedding_id];

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = embedding_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        std::vector<emb_t> grad_ev;

        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          std::copy_n(top_grad[gpu_id].begin() + ev_offset * batch_size_per_gpu + local_b * ev_size,
                      ev_size, std::back_inserter(grad_ev));
        } else {
          std::copy_n(top_grad[gpu_id].begin() + ev_offset * batch_size_per_gpu +
                          local_b * hotness * ev_size,
                      hotness * ev_size, std::back_inserter(grad_ev));
        }

        if (combiner == Combiner::Average) {
          for (size_t i = 0; i < grad_ev.size(); ++i) {
            float gi = HugeCTR::TypeConvert<float, emb_t>::convert(grad_ev[i]);
            grad_ev[i] = HugeCTR::TypeConvert<emb_t, float>::convert(gi / (end - start));
          }
        }
        for (int r = 0; r < static_cast<int>(end - start); ++r) {
          key_t k = keys[start + r];
          if (accumulate_grad_map_[id_space].find(k) == accumulate_grad_map_[id_space].end()) {
            accumulate_grad_map_[id_space][k].assign(ev_size, 0.f);
          }
          for (int e = 0; e < ev_size; ++e) {
            float gi;
            if (combiner == Combiner::Sum || combiner == Combiner::Average) {
              gi = HugeCTR::TypeConvert<float, emb_t>::convert(grad_ev[e]);
            } else {
              gi = HugeCTR::TypeConvert<float, emb_t>::convert(grad_ev[r * ev_size + e]);
            }
            accumulate_grad_map_[id_space][k][e] += gi;
          }
        }
      }
    }
  }

  void embedding_update_cpu() { emb_table_cpu_.update(accumulate_grad_map_); }
};