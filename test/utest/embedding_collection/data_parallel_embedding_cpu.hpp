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

#include <embedding/view.hpp>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <utest/embedding_collection/embedding_collection_utils.hpp>
#include <utest/embedding_collection/embedding_table_cpu.hpp>

using namespace embedding;

struct DataParallelEmbeddingMetaCPU {
  int gpu_id;
  int num_gpus;

  int num_lookup;
  std::vector<int> table_ids;
  std::vector<int> lookup_ids;
  std::vector<int> max_hotnesses;
  std::vector<Combiner> combiners;
  std::vector<int> ev_sizes;
  std::vector<int> ev_offsets;

  std::vector<int> dst_buffer_ev_sizes;
  std::vector<int> dst_buffer_ev_offsets;
  DataParallelEmbeddingMetaCPU(int _gpu_id, int _num_gpus,
                               const EmbeddingCollectionParam &ebc_param, int emb_id)
      : gpu_id(_gpu_id), num_gpus(_num_gpus), ev_offsets{0}, dst_buffer_ev_offsets{0} {
    const auto &emb_param = ebc_param.grouped_emb_params[emb_id];
    for (int lookup_id = 0; lookup_id < static_cast<int>(ebc_param.lookup_params.size());
         ++lookup_id) {
      const auto &lookup_param = ebc_param.lookup_params[lookup_id];
      Combiner combiner = lookup_param.combiner;
      int max_hotness = lookup_param.max_hotness;
      int ev_size = (combiner == Combiner::Concat) ? max_hotness * lookup_param.ev_size
                                                   : lookup_param.ev_size;
      dst_buffer_ev_sizes.push_back(ev_size);
      dst_buffer_ev_offsets.push_back(ev_size);

      int table_id = lookup_param.table_id;
      if (std::find(emb_param.table_ids.begin(), emb_param.table_ids.end(), table_id) ==
          emb_param.table_ids.end()) {
        continue;
      }
      HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                      "dp table must be shared on all gpus");
      table_ids.push_back(table_id);
      lookup_ids.push_back(lookup_id);
      max_hotnesses.push_back(lookup_param.max_hotness);
      combiners.push_back(lookup_param.combiner);
      HCTR_CHECK_HINT(lookup_param.combiner != Combiner::Concat,
                      "DataParallelEmbedding CPU does not support concat combiner");
      ev_sizes.push_back(lookup_param.ev_size);
      ev_offsets.push_back(lookup_param.ev_size);
    }
    num_lookup = static_cast<int>(lookup_ids.size());

    std::partial_sum(ev_offsets.begin(), ev_offsets.end(), ev_offsets.begin());
    std::partial_sum(dst_buffer_ev_offsets.begin(), dst_buffer_ev_offsets.end(),
                     dst_buffer_ev_offsets.begin());
  }
};

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class DataParallelEmbeddingCPU {
 public:
  int num_gpus_;
  std::vector<DataParallelEmbeddingMetaCPU> metas_;
  std::vector<std::vector<key_t>> selected_keys_;
  std::vector<std::vector<uint32_t>> num_selected_keys_per_bucket_offset_;

  DataParallelEmbeddingCPU(int num_gpus, const EmbeddingCollectionParam &ebc_param, int emb_id)
      : num_gpus_(num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      metas_.emplace_back(gpu_id, num_gpus, ebc_param, emb_id);
    }
    selected_keys_.resize(num_gpus_);
    num_selected_keys_per_bucket_offset_.resize(num_gpus_);
  }

  void forward_per_gpu(int gpu_id, const std::vector<key_t> &keys,
                       const std::vector<offset_t> &bucket_range,
                       const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                       std::vector<emb_t> &embedding_vec, int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    selected_keys_[gpu_id].clear();
    num_selected_keys_per_bucket_offset_[gpu_id].clear();
    num_selected_keys_per_bucket_offset_[gpu_id].push_back(0);
    ASSERT_TRUE(gpu_id == metas_[gpu_id].gpu_id);

    for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
      int lookup_id = metas_[gpu_id].lookup_ids[i];
      for (int b = 0; b < batch_size_per_gpu; ++b) {
        int bucket_id = batch_size * lookup_id + gpu_id * batch_size_per_gpu + b;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);
        for (uint32_t r = 0; r < end - start; ++r) {
          selected_keys_[gpu_id].push_back(keys[start + r]);
        }
        num_selected_keys_per_bucket_offset_[gpu_id].push_back(end - start);
      }
    }
    std::partial_sum(num_selected_keys_per_bucket_offset_[gpu_id].begin(),
                     num_selected_keys_per_bucket_offset_[gpu_id].end(),
                     num_selected_keys_per_bucket_offset_[gpu_id].begin());

    for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
      int table_id = metas_[gpu_id].table_ids[i];
      int lookup_id = metas_[gpu_id].lookup_ids[i];
      int ev_size = metas_[gpu_id].ev_sizes[i];
      Combiner combiner = metas_[gpu_id].combiners[i];

      int dst_buffer_ev_offset = metas_[gpu_id].dst_buffer_ev_offsets[lookup_id];
      int dst_buffer_ev_size = metas_[gpu_id].dst_buffer_ev_sizes[lookup_id];
      ASSERT_EQ(dst_buffer_ev_size, ev_size);

      for (int b = 0; b < batch_size_per_gpu; ++b) {
        int i_bucket = batch_size_per_gpu * i + b;
        uint32_t start = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket];
        uint32_t end = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket + 1];

        ArrayView<emb_t> dst_ev{
            &embedding_vec[batch_size_per_gpu * dst_buffer_ev_offset + b * ev_size], ev_size};
        ASSERT_EQ(dst_ev.size(), metas_[gpu_id].ev_sizes[i]);
        std::vector<float> accumulate_vec;
        accumulate_vec.assign(dst_ev.size(), 0.f);

        for (uint32_t r = 0; r < (end - start); ++r) {
          key_t k = selected_keys_[gpu_id][start + r];
          ASSERT_TRUE(table_id < static_cast<int>(emb_table_cpu.emb_table_list_.size()));
          ASSERT_TRUE(emb_table_cpu.emb_table_list_[table_id].find(k) !=
                      emb_table_cpu.emb_table_list_[table_id].end());
          auto ev = emb_table_cpu.emb_table_list_[table_id].at(k);
          ASSERT_TRUE(ev.size() == accumulate_vec.size());
          for (int e = 0; e < dst_ev.size(); ++e) {
            accumulate_vec[e] += ev[e];
          }
        }

        if (combiner == Combiner::Average && (end - start) > 0) {
          for (int i = 0; i < dst_ev.size(); ++i) {
            accumulate_vec[i] /= (end - start);
          }
        }

        for (int e = 0; e < dst_ev.size(); ++e) {
          dst_ev[e] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[e]);
        }
      }
    }
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range,
                             const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                             std::vector<std::vector<emb_t>> &embedding_vec, int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      forward_per_gpu(gpu_id, keys, bucket_range, emb_table_cpu, embedding_vec[gpu_id], batch_size);
    }
  }

  void backward_per_gpu(int gpu_id, const std::vector<emb_t> &top_grad,
                        std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info,
                        int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    std::vector<float> float_top_grad;
    float_top_grad.resize(batch_size_per_gpu * metas_[gpu_id].ev_offsets.back());
    for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
      int lookup_id = metas_[gpu_id].lookup_ids[i];
      int ev_size = metas_[gpu_id].ev_sizes[i];
      int ev_offset = metas_[gpu_id].ev_offsets[i];
      Combiner combiner = metas_[gpu_id].combiners[i];

      int dst_buffer_ev_offset = metas_[gpu_id].dst_buffer_ev_offsets[lookup_id];
      int dst_buffer_ev_size = metas_[gpu_id].dst_buffer_ev_sizes[lookup_id];
      ASSERT_EQ(dst_buffer_ev_size, ev_size);

      for (int b = 0; b < batch_size_per_gpu; ++b) {
        ArrayView<float> float_top_grad_ev{
            &float_top_grad[batch_size_per_gpu * ev_offset + b * ev_size], ev_size};
        ArrayView<emb_t> top_grad_ev{
            const_cast<emb_t *>(&top_grad[batch_size_per_gpu * dst_buffer_ev_offset + b * ev_size]),
            ev_size};
        int i_bucket = i * batch_size_per_gpu + b;
        int num_keys = (combiner == Combiner::Average)
                           ? num_selected_keys_per_bucket_offset_[gpu_id][i_bucket + 1] -
                                 num_selected_keys_per_bucket_offset_[gpu_id][i_bucket]
                           : 1;
        for (int e = 0; e < float_top_grad_ev.size(); ++e) {
          if (num_keys > 0) {
            float_top_grad_ev[e] =
                HugeCTR::TypeConvert<float, emb_t>::convert(top_grad_ev[e]) / num_keys;
          }
        }
      }
    }

    std::vector<std::unordered_map<key_t, std::vector<std::vector<float>>>> local_reduce_grad;
    local_reduce_grad.resize(grad_info.size());
    for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
      int table_id = metas_[gpu_id].table_ids[i];
      ASSERT_TRUE(table_id < static_cast<int>(grad_info.size()));
      auto &wgrad_dict = local_reduce_grad[table_id];
      int ev_size = metas_[gpu_id].ev_sizes[i];
      int ev_offset = metas_[gpu_id].ev_offsets[i];

      for (int b = 0; b < batch_size_per_gpu; ++b) {
        int i_bucket = i * batch_size_per_gpu + b;
        uint32_t start = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket];
        uint32_t end = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket + 1];

        ArrayView<float> float_ev{&float_top_grad[batch_size_per_gpu * ev_offset + b * ev_size],
                                  ev_size};
        for (uint32_t r = 0; r < end - start; ++r) {
          key_t k = selected_keys_[gpu_id][r + start];
          std::vector<float> evs;
          for (int e = 0; e < float_ev.size(); ++e) {
            evs.push_back(HugeCTR::TypeConvert<float, emb_t>::convert(float_ev[e]));
          }
          ASSERT_EQ(evs.size(), ev_size);
          wgrad_dict[k].push_back(evs);
        }
      }
    }

    for (size_t table_id = 0; table_id < local_reduce_grad.size(); ++table_id) {
      auto &table_grad = local_reduce_grad[table_id];
      for (auto &[k, evs] : table_grad) {
        HCTR_CHECK(evs.size() >= 1);
        int ev_size = evs[0].size();
        std::vector<float> ev_sum;
        for (int e = 0; e < ev_size; ++e) {
          std::vector<float> arr;
          for (size_t i = 0; i < evs.size(); ++i) {
            arr.push_back(evs[i][e]);
          }
          float summation = kahanSum(arr);
          ev_sum.push_back(summation);
        }
        if (grad_info[table_id].find(k) == grad_info[table_id].end()) {
          grad_info[table_id][k].assign(ev_size, 0);
        }
        for (int e = 0; e < ev_size; ++e) {
          grad_info[table_id][k][e] += ev_sum[e];
        }
      }
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grads,
                              std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info,
                              int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      backward_per_gpu(gpu_id, top_grads[gpu_id], grad_info, batch_size);
    }
  }
};