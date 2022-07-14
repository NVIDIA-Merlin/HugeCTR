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

#include "embedding_table_cpu.hpp"
#include "view_cpu.hpp"

using namespace embedding;

template <typename key_t, typename offset_t>
struct DPForwardIndicesCPU {
  std::vector<key_t> dp_key_list_;
  std::vector<uint32_t> dp_offset_list_;
  std::vector<int> dp_dst_list_;

  int num_local_embedding_;
  std::vector<int> local_embedding_list_;
  int gpu_id_;
  int num_gpus_;

  DPForwardIndicesCPU(const std::vector<int> &local_embedding_list, int gpu_id, int num_gpu)
      : num_local_embedding_(local_embedding_list.size()),
        local_embedding_list_(local_embedding_list),
        gpu_id_(gpu_id),
        num_gpus_(num_gpu) {}

  void compute(const std::vector<key_t> &keys, const std::vector<offset_t> &bucket_range,
               int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    dp_key_list_.clear();
    dp_offset_list_.clear();
    dp_offset_list_.push_back(0);
    dp_dst_list_.clear();
    for (int tid = 0; tid < num_local_embedding_ * batch_size_per_gpu; ++tid) {
      int embedding_id = local_embedding_list_[tid / batch_size_per_gpu];
      int batch_id = tid % batch_size_per_gpu;

      int bucket_id = batch_size * embedding_id + gpu_id_ * batch_size_per_gpu + batch_id;
      offset_t start = bucket_range[bucket_id];
      offset_t end = bucket_range[bucket_id + 1];
      for (offset_t i = 0; i < (end - start); ++i) {
        dp_key_list_.push_back(keys[start + i]);
      }
      dp_offset_list_.push_back(static_cast<uint32_t>(end - start));

      int dst_bucket_id = batch_size_per_gpu * embedding_id + batch_id;
      dp_dst_list_.push_back(dst_bucket_id);
    }

    std::inclusive_scan(dp_offset_list_.begin(), dp_offset_list_.end(), dp_offset_list_.begin());
  }
};

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class DataParallelEmbeddingCPU {
 public:
  int num_gpus_;
  int num_embedding_;
  int num_table_;
  EmbeddingCollectionParam ebc_param_;
  std::vector<EmbeddingShardingParam> sharding_param_list_;

  std::vector<int> ev_size_list_;
  std::vector<int> ev_offset_list_;

  std::vector<std::vector<int>> local_id_space_list_;
  std::vector<std::vector<int>> local_embedding_list_;
  std::vector<std::vector<int>> local_hotness_list_;
  std::vector<std::vector<int>> local_ev_size_list_;
  std::vector<std::vector<int>> local_ev_offset_list_;
  std::vector<std::vector<char>> local_combiner_list_;

  std::vector<DPForwardIndicesCPU<key_t, offset_t>> forward_indices_;

  std::vector<std::vector<key_t>> unique_key_list_;
  std::vector<std::vector<uint32_t>> sorted_bucket_id_list_;
  std::vector<std::vector<uint32_t>> sorted_bucket_id_offset_list_;
  std::vector<std::vector<int>> num_unique_key_scan_list_;

  std::vector<RaggedGradBufferCPU<float>> grad_;

  std::vector<std::unordered_map<key_t, std::vector<uint32_t>>> dp_backward_info_;

  DataParallelEmbeddingCPU(int num_gpus, int num_table, const EmbeddingCollectionParam &ebc_param,
                           const std::vector<EmbeddingShardingParam> &sharding_param_list)
      : num_gpus_(num_gpus),
        num_embedding_(ebc_param.num_embedding),
        num_table_(num_table),
        ebc_param_(ebc_param),
        sharding_param_list_(sharding_param_list),
        ev_offset_list_{0} {
    for (int i = 0; i < num_embedding_; ++i) {
      ev_size_list_.push_back(ebc_param_.embedding_params[i].ev_size);
      ev_offset_list_.push_back(ebc_param_.embedding_params[i].ev_size);
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());

    local_id_space_list_.resize(num_gpus_);
    local_embedding_list_.resize(num_gpus_);
    local_hotness_list_.resize(num_gpus_);
    local_ev_size_list_.resize(num_gpus_);
    local_ev_offset_list_.resize(num_gpus_);
    local_combiner_list_.resize(num_gpus_);

    auto &embedding_params = ebc_param_.embedding_params;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      local_ev_offset_list_[gpu_id].push_back(0);
      for (int embedding_id : sharding_param_list[gpu_id].local_embedding_list) {
        local_id_space_list_[gpu_id].push_back(embedding_params[embedding_id].id_space);
        local_embedding_list_[gpu_id].push_back(embedding_id);
        local_hotness_list_[gpu_id].push_back(embedding_params[embedding_id].hotness);
        local_ev_size_list_[gpu_id].push_back(embedding_params[embedding_id].ev_size);
        local_ev_offset_list_[gpu_id].push_back(embedding_params[embedding_id].ev_size);
        local_combiner_list_[gpu_id].push_back(
            static_cast<char>(embedding_params[embedding_id].combiner));
      }
      std::inclusive_scan(local_ev_offset_list_[gpu_id].begin(),
                          local_ev_offset_list_[gpu_id].end(),
                          local_ev_offset_list_[gpu_id].begin());
    }

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      forward_indices_.emplace_back(local_embedding_list_[gpu_id], gpu_id, num_gpus_);
    }

    unique_key_list_.resize(num_gpus);
    sorted_bucket_id_list_.resize(num_gpus);
    sorted_bucket_id_offset_list_.resize(num_gpus);
    num_unique_key_scan_list_.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      grad_.emplace_back(ebc_param.universal_batch_size, local_id_space_list_[gpu_id],
                         local_ev_size_list_[gpu_id], local_hotness_list_[gpu_id]);
    }
  }

  void cpu_dp_model_forward_per_gpu(int gpu_id, int batch_size,
                                    const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                                    std::vector<std::vector<emb_t>> &embedding_vec) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    RaggedEmbForwardResultViewCPU<emb_t> forward_result_view{&embedding_vec[gpu_id], &ev_size_list_,
                                                             &ev_offset_list_, batch_size_per_gpu};

    auto &indices = forward_indices_[gpu_id];
    for (int bucket_id = 0; bucket_id < static_cast<int>(indices.dp_offset_list_.size()) - 1;
         ++bucket_id) {
      int local_emb_id = bucket_id / batch_size_per_gpu;

      int id_space = local_id_space_list_[gpu_id][local_emb_id];
      char combiner = local_combiner_list_[gpu_id][local_emb_id];

      int dst_bucket_id = indices.dp_dst_list_[bucket_id];
      ArrayView<emb_t> dst_ev = forward_result_view[dst_bucket_id];
      std::vector<float> accumulate_vec;
      accumulate_vec.assign(dst_ev.size(), 0.f);

      uint32_t start = indices.dp_offset_list_[bucket_id];
      uint32_t end = indices.dp_offset_list_[bucket_id + 1];
      for (uint32_t r = 0; r < (end - start); ++r) {
        key_t k = indices.dp_key_list_[start + r];
        ASSERT_TRUE(emb_table_cpu.emb_table_list_[id_space].find(k) !=
                    emb_table_cpu.emb_table_list_[id_space].end());
        auto ev = emb_table_cpu.emb_table_list_[id_space].at(k);
        assert(ev.size() == accumulate_vec.size());

        for (int i = 0; i < dst_ev.size(); ++i) {
          accumulate_vec[i] += ev[i];
        }
      }

      if (combiner == static_cast<char>(Combiner::Average) && (end - start) > 0) {
        for (int i = 0; i < dst_ev.size(); ++i) {
          accumulate_vec[i] /= (end - start);
        }
      }

      for (int i = 0; i < dst_ev.size(); ++i) {
        dst_ev[i] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[i]);
      }
    }
  }

  void cpu_dp_backward_index_calculation_per_gpu(int gpu_id, int batch_size,
                                                 const std::vector<key_t> &keys,
                                                 const std::vector<offset_t> &bucket_range) {
    if (gpu_id == 0) {
      dp_backward_info_.resize(local_id_space_list_[gpu_id].size());
    }
    int batch_size_per_gpu = batch_size / num_gpus_;

    for (size_t i = 0; i < local_embedding_list_[gpu_id].size(); ++i) {
      int embedding_id = local_embedding_list_[gpu_id][i];

      int batch_start = gpu_id * batch_size_per_gpu;
      for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
        uint32_t bucket_id = embedding_id * batch_size + batch_start + batch_id;

        uint32_t start = bucket_range[bucket_id];
        uint32_t end = bucket_range[bucket_id + 1];
        for (uint32_t r = 0; r < (end - start); ++r) {
          dp_backward_info_[i][keys[r + start]].push_back(bucket_id);
        }
      }
    }
  }

  void cpu_dp_local_reduce_per_gpu(
      int gpu_id, const std::vector<std::vector<emb_t>> &top_grads,
      std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info, int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    std::vector<std::unordered_map<key_t, std::vector<float>>> local_reduce_grad_info;
    local_reduce_grad_info.resize(grad_info.size());

    for (size_t idx = 0; idx < local_id_space_list_[gpu_id].size(); ++idx) {
      int id_space = local_id_space_list_[gpu_id][idx];
      int ev_size = local_ev_size_list_[gpu_id][idx];

      auto &dp_backward_info_in_current_id_space = dp_backward_info_[idx];
      auto &grad_info_in_current_id_space = local_reduce_grad_info[id_space];
      for (auto &key_bucket_id_pair : dp_backward_info_in_current_id_space) {
        key_t k = key_bucket_id_pair.first;
        auto &bucket_id_list = key_bucket_id_pair.second;
        for (auto bucket_id : bucket_id_list) {
          int embedding_id = bucket_id / batch_size;
          int batch_id = bucket_id % batch_size;
          if (batch_id >= gpu_id * batch_size_per_gpu &&
              batch_id < (gpu_id + 1) * batch_size_per_gpu) {
            int local_batch_id = batch_id - gpu_id * batch_size_per_gpu;
            std::vector<float> gi;
            int start =
                ev_offset_list_[embedding_id] * batch_size_per_gpu + local_batch_id * ev_size;
            int end = start + ev_size;
            for (int i = start; i < end; ++i) {
              gi.push_back(HugeCTR::TypeConvert<float, emb_t>::convert(top_grads[gpu_id][i]));
            }
            for (size_t i = 0; i < gi.size(); ++i) {
              if (grad_info_in_current_id_space.find(k) == grad_info_in_current_id_space.end()) {
                grad_info_in_current_id_space[k].assign(ev_size, 0.f);
              }
              grad_info_in_current_id_space[k][i] += gi[i];
            }
          }
        }
      }
    }
    for (size_t idx = 0; idx < local_id_space_list_[gpu_id].size(); ++idx) {
      int id_space = local_id_space_list_[gpu_id][idx];
      int ev_size = local_ev_size_list_[gpu_id][idx];

      auto &local_grad_in_current_id_space = local_reduce_grad_info[id_space];
      auto &grad_info_in_current_id_space = grad_info[id_space];
      for (auto &[k, ev] : local_grad_in_current_id_space) {
        if (grad_info_in_current_id_space.find(k) == grad_info_in_current_id_space.end()) {
          grad_info_in_current_id_space[k].assign(ev_size, 0.f);
        }
        ASSERT_EQ(grad_info_in_current_id_space[k].size(), ev_size);
        ASSERT_EQ(local_grad_in_current_id_space[k].size(), ev_size);
        for (int i = 0; i < ev_size; ++i) {
          grad_info_in_current_id_space[k][i] += local_grad_in_current_id_space[k][i];
        }
      }
    }

    // for (size_t idx = 0; idx < local_id_space_list_[gpu_id].size(); ++idx) {
    //   int id_space = local_id_space_list_[gpu_id][idx];
    //   int ev_size = local_ev_size_list_[gpu_id][idx];

    //   auto &dp_backward_info_in_current_id_space = dp_backward_info_[idx];
    //   auto &grad_info_in_current_id_space = grad_info[id_space];
    //   for (auto &key_bucket_id_pair : dp_backward_info_in_current_id_space) {
    //     key_t k = key_bucket_id_pair.first;
    //     auto &bucket_id_list = key_bucket_id_pair.second;
    //     for (auto bucket_id : bucket_id_list) {
    //       int embedding_id = bucket_id / batch_size;
    //       int batch_id = bucket_id % batch_size;
    //       if (batch_id >= gpu_id * batch_size_per_gpu &&
    //           batch_id < (gpu_id + 1) * batch_size_per_gpu) {
    //         int local_batch_id = batch_id - gpu_id * batch_size_per_gpu;
    //         std::vector<float> gi;
    //         int start =
    //             ev_offset_list_[embedding_id] * batch_size_per_gpu + local_batch_id * ev_size;
    //         int end = start + ev_size;
    //         for (int i = start; i < end; ++i) {
    //           gi.push_back(HugeCTR::TypeConvert<float, emb_t>::convert(top_grads[gpu_id][i]));
    //         }
    //         for (size_t i = 0; i < gi.size(); ++i) {
    //           if (grad_info_in_current_id_space.find(k) == grad_info_in_current_id_space.end()) {
    //             grad_info_in_current_id_space[k].assign(ev_size, 0.f);
    //           }
    //           grad_info_in_current_id_space[k][i] += gi[i];
    //         }
    //       }
    //     }
    //   }
    // }
  }

  void embedding_forward_cpu(const std::vector<key_t> &t_keys,
                             const std::vector<offset_t> &flatten_concat_bucket_range,
                             const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                             std::vector<std::vector<emb_t>> &embedding_vec, int batch_size) {
    dp_backward_info_.clear();
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      forward_indices_[gpu_id].compute(t_keys, flatten_concat_bucket_range, batch_size);
    }

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_dp_model_forward_per_gpu(gpu_id, batch_size, emb_table_cpu, embedding_vec);
      cpu_dp_backward_index_calculation_per_gpu(gpu_id, batch_size, t_keys,
                                                flatten_concat_bucket_range);
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grads,
                              std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info,
                              int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_dp_local_reduce_per_gpu(gpu_id, top_grads, grad_info, batch_size);
    }
  }
};