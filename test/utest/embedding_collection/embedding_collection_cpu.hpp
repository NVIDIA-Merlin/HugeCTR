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
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <utest/embedding_collection/data_parallel_embedding_cpu.hpp>
#include <utest/embedding_collection/model_parallel_embedding_cpu.hpp>
#include <utest/embedding_collection/reference_embedding.hpp>

using namespace embedding;

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class EmbeddingCollectionCPU {
 public:
  int num_gpus_;
  EmbeddingCollectionParam ebc_param_;
  int num_table_;

  std::vector<std::vector<emb_t>> embedding_vec_;
  std::vector<std::unordered_map<key_t, std::vector<float>>> grad_info_;

  std::vector<ModelParallelEmbeddingCPU<key_t, offset_t, index_t, emb_t>> mp_embedding_list_;
  std::vector<DataParallelEmbeddingCPU<key_t, offset_t, index_t, emb_t>> dp_embedding_list_;

  EmbeddingTableCPU<key_t, index_t> emb_table_cpu_;

  EmbeddingCollectionCPU(int num_gpus, const EmbeddingCollectionParam &ebc_param, int num_table,
                         const std::vector<EmbeddingTableParam> &table_param_list,
                         std::vector<std::vector<IGroupedEmbeddingTable *>> emb_table_list)
      : num_gpus_(num_gpus),
        ebc_param_(ebc_param),
        num_table_(num_table),
        emb_table_cpu_(num_table, emb_table_list, table_param_list) {
    for (size_t emb_id = 0; emb_id < ebc_param.grouped_emb_params.size(); ++emb_id) {
      if (ebc_param.grouped_emb_params[emb_id].table_placement_strategy ==
          TablePlacementStrategy::DataParallel) {
        dp_embedding_list_.emplace_back(num_gpus, ebc_param, emb_id);
      } else if (ebc_param.grouped_emb_params[emb_id].table_placement_strategy ==
                 TablePlacementStrategy::ModelParallel) {
        mp_embedding_list_.emplace_back(num_gpus, ebc_param, emb_id);
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "EmbeddingCollectionCPU does not support table placement strategy");
      }
    }
    grad_info_.resize(num_table_);

    embedding_vec_.resize(num_gpus_);
    int num_ev_elems = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      num_ev_elems += ebc_param.lookup_params[lookup_id].ev_size;
    }
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      embedding_vec_[gpu_id].resize(num_ev_elems * ebc_param.universal_batch_size / num_gpus_);
    }
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / ebc_param_.num_lookup;

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      for (size_t i = 0; i < embedding_vec_[gpu_id].size(); ++i) {
        embedding_vec_[gpu_id][i] = HugeCTR::TypeConvert<emb_t, float>::convert(0.f);
      }
    }

    for (auto &embedding : mp_embedding_list_) {
      embedding.embedding_forward_cpu(keys, bucket_range, emb_table_cpu_, embedding_vec_,
                                      batch_size);
    }
    for (auto &embedding : dp_embedding_list_) {
      embedding.embedding_forward_cpu(keys, bucket_range, emb_table_cpu_, embedding_vec_,
                                      batch_size);
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grads, int batch_size) {
    for (auto &grad : grad_info_) {
      grad.clear();
    }

    for (auto &embedding : mp_embedding_list_) {
      embedding.embedding_backward_cpu(top_grads, grad_info_, batch_size);
    }
    for (auto &embedding : dp_embedding_list_) {
      embedding.embedding_backward_cpu(top_grads, grad_info_, batch_size);
    }
  }

  void embedding_update_cpu() { emb_table_cpu_.update(grad_info_); }
};
