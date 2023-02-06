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
#include "embedding_collection_utils.hpp"
#include "embedding_table_cpu.hpp"
using namespace embedding;

struct ModelParallelEmbeddingMetaCPU {
  int gpu_id;
  int num_gpus;

  int num_lookup;
  std::vector<int> shard_ids;
  std::vector<int> num_shards;
  std::vector<int> table_ids;
  std::vector<int> lookup_ids;
  std::vector<int> max_hotnesses;
  std::vector<int> ev_sizes;
  std::vector<int> ev_offsets;

  // network
  std::vector<std::vector<int>> network_ev_sizes;
  std::vector<std::vector<int>> network_ev_offsets;

  std::vector<int> network_ids;      // 0, 1
  std::vector<int> network_gpu_ids;  // 0, 1
  std::vector<int> network_offsets;  // 0, 2

  int num_network_dst_look;
  std::vector<int> network_dst_lookup_ids;
  std::vector<Combiner> network_dst_combiners;

  std::vector<int> dst_buffer_ev_sizes;
  std::vector<int> dst_buffer_ev_offsets;

  ModelParallelEmbeddingMetaCPU(int _gpu_id, int _num_gpus,
                                const EmbeddingCollectionParam &ebc_param, int emb_id)
      : gpu_id(_gpu_id), num_gpus(_num_gpus), ev_offsets{0}, dst_buffer_ev_offsets{0} {
    const auto &emb_param = ebc_param.grouped_emb_params[emb_id];
    for (int lookup_id = 0; lookup_id < static_cast<int>(ebc_param.lookup_params.size());
         ++lookup_id) {
      const auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int table_id = lookup_param.table_id;
      Combiner combiner = lookup_param.combiner;
      int max_hotness = lookup_param.max_hotness;
      int ev_size = (combiner == Combiner::Concat) ? max_hotness * lookup_param.ev_size
                                                   : lookup_param.ev_size;
      dst_buffer_ev_sizes.push_back(ev_size);
      dst_buffer_ev_offsets.push_back(ev_size);

      if (std::find(emb_param.table_ids.begin(), emb_param.table_ids.end(), table_id) ==
          emb_param.table_ids.end()) {
        continue;
      }
      if (ebc_param.shard_matrix[gpu_id][table_id] == 0) {
        continue;
      }
      HCTR_CHECK_HINT(combiner != Combiner::Concat,
                      "ModelParallelEmbedding CPU does not support concat combiner");

      std::vector<int> shard_gpus;
      for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
        if (ebc_param.shard_matrix[ggpu_id][table_id] == 1) {
          shard_gpus.push_back(ggpu_id);
        }
      }
      auto find_shard_id_iter = std::find(shard_gpus.begin(), shard_gpus.end(), gpu_id);
      HCTR_CHECK_HINT(find_shard_id_iter != shard_gpus.end(),
                      "ModelParallelEmbedding CPU does not find shard id");
      shard_ids.push_back(std::distance(shard_gpus.begin(), find_shard_id_iter));
      num_shards.push_back(static_cast<int>(shard_gpus.size()));
      table_ids.push_back(table_id);
      lookup_ids.push_back(lookup_id);
      max_hotnesses.push_back(lookup_param.max_hotness);
      ev_sizes.push_back(lookup_param.ev_size);
      ev_offsets.push_back(lookup_param.ev_size);
    }
    num_lookup = static_cast<int>(lookup_ids.size());

    std::partial_sum(ev_offsets.begin(), ev_offsets.end(), ev_offsets.begin());
    std::partial_sum(dst_buffer_ev_offsets.begin(), dst_buffer_ev_offsets.end(),
                     dst_buffer_ev_offsets.begin());

    // network
    network_ev_sizes.resize(num_gpus);
    network_ev_offsets.resize(num_gpus);
    for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
      network_ev_offsets[ggpu_id].push_back(0);
    }

    std::vector<std::tuple<int, int, int>> network_buffer_meta_info;
    for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
      int network_id = 0;
      for (int lookup_id = 0; lookup_id < static_cast<int>(ebc_param.lookup_params.size());
           ++lookup_id) {
        const auto &lookup_param = ebc_param.lookup_params[lookup_id];
        int table_id = lookup_param.table_id;
        int ev_size = lookup_param.ev_size;

        if (std::find(emb_param.table_ids.begin(), emb_param.table_ids.end(), table_id) ==
            emb_param.table_ids.end()) {
          continue;
        }
        if (ebc_param.shard_matrix[ggpu_id][table_id] == 0) {
          continue;
        }
        network_ev_sizes[ggpu_id].push_back(ev_size);
        network_ev_offsets[ggpu_id].push_back(ev_size);
        network_buffer_meta_info.push_back({ggpu_id, network_id, lookup_id});
        network_id += 1;
      }
    }
    for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
      std::partial_sum(network_ev_offsets[ggpu_id].begin(), network_ev_offsets[ggpu_id].end(),
                       network_ev_offsets[ggpu_id].begin());
    }

    std::sort(
        network_buffer_meta_info.begin(), network_buffer_meta_info.end(),
        [](const auto &lhs, const auto &rhs) { return std::get<2>(lhs) <= std::get<2>(rhs); });

    for (size_t i = 0; i < network_buffer_meta_info.size(); ++i) {
      const auto &meta_info = network_buffer_meta_info[i];
      int network_gpu_id = std::get<0>(meta_info);
      int network_id = std::get<1>(meta_info);
      network_ids.push_back(network_id);
      network_gpu_ids.push_back(network_gpu_id);
    }

    for (size_t i = 0; i < network_buffer_meta_info.size(); ++i) {
      const auto &meta_info = network_buffer_meta_info[i];
      int lookup_id = std::get<2>(meta_info);
      Combiner combiner = ebc_param.lookup_params[lookup_id].combiner;
      if (i == 0 || lookup_id != std::get<2>(network_buffer_meta_info[i - 1])) {
        network_dst_lookup_ids.push_back(lookup_id);
        network_dst_combiners.push_back(combiner);
      }
    }
    num_network_dst_look = static_cast<int>(network_dst_lookup_ids.size());

    int network_offset = 0;
    for (size_t i = 0; i < network_buffer_meta_info.size(); ++i) {
      const auto &meta_info = network_buffer_meta_info[i];
      int lookup_id = std::get<2>(meta_info);
      if (i == 0 || lookup_id != std::get<2>(network_buffer_meta_info[i - 1])) {
        network_offsets.push_back(network_offset);
      }
      network_offset += 1;
    }
    network_offsets.push_back(network_offset);
  }
};

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class ModelParallelEmbeddingCPU {
  int num_gpus_;
  std::vector<ModelParallelEmbeddingMetaCPU> metas_;
  std::vector<std::vector<key_t>> selected_keys_;
  std::vector<std::vector<uint32_t>> num_selected_keys_per_bucket_offset_;
  std::vector<std::vector<uint32_t>> pooling_factor_per_bucket_;

  std::vector<std::vector<std::vector<emb_t>>> model_comm_buffers_;
  std::vector<std::vector<std::vector<emb_t>>> network_comm_buffers_;

 public:
  ModelParallelEmbeddingCPU(int num_gpus, const EmbeddingCollectionParam &ebc_param, int emb_id)
      : num_gpus_(num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      metas_.emplace_back(gpu_id, num_gpus, ebc_param, emb_id);
    }
    selected_keys_.resize(num_gpus_);
    num_selected_keys_per_bucket_offset_.resize(num_gpus_);
    pooling_factor_per_bucket_.resize(num_gpus_);

    model_comm_buffers_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      model_comm_buffers_[gpu_id].resize(num_gpus_);
      for (int ggpu_id = 0; ggpu_id < num_gpus_; ++ggpu_id) {
        model_comm_buffers_[gpu_id][ggpu_id].resize(metas_[gpu_id].ev_offsets.back() *
                                                    ebc_param.universal_batch_size / num_gpus_);
      }
    }
    network_comm_buffers_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      network_comm_buffers_[gpu_id].resize(num_gpus_);
      for (int ggpu_id = 0; ggpu_id < num_gpus_; ++ggpu_id) {
        network_comm_buffers_[gpu_id][ggpu_id].resize(
            metas_[gpu_id].network_ev_offsets[ggpu_id].back() * ebc_param.universal_batch_size /
            num_gpus_);
      }
    }
  }

  void all2all(const std::vector<std::vector<std::vector<emb_t>>> &send_buffer,
               std::vector<std::vector<std::vector<emb_t>>> &recv_buffer) {
    for (int src_gpu_id = 0; src_gpu_id < num_gpus_; ++src_gpu_id) {
      for (int dst_gpu_id = 0; dst_gpu_id < num_gpus_; ++dst_gpu_id) {
        auto &send_tensor = send_buffer[src_gpu_id][dst_gpu_id];
        auto &recv_tensor = recv_buffer[dst_gpu_id][src_gpu_id];
        ASSERT_EQ(send_tensor.size(), recv_tensor.size());
        recv_tensor = send_tensor;
      }
    }
  }

  void model_forward_per_gpu(int gpu_id, const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range,
                             const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                             int batch_size) {
    selected_keys_[gpu_id].clear();
    num_selected_keys_per_bucket_offset_[gpu_id].clear();
    num_selected_keys_per_bucket_offset_[gpu_id].push_back(0);
    pooling_factor_per_bucket_[gpu_id].clear();
    ASSERT_TRUE(gpu_id == metas_[gpu_id].gpu_id);

    for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
      int lookup_id = metas_[gpu_id].lookup_ids[i];
      int shard_id = metas_[gpu_id].shard_ids[i];
      int num_shard = metas_[gpu_id].num_shards[i];

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = batch_size * lookup_id + b;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);
        uint32_t num_selected_key = 0;
        for (uint32_t r = 0; r < end - start; ++r) {
          key_t k = keys[start + r];
          if (k % num_shard == shard_id) {
            selected_keys_[gpu_id].push_back(keys[start + r]);
            num_selected_key += 1;
          }
        }
        num_selected_keys_per_bucket_offset_[gpu_id].push_back(num_selected_key);
      }
    }
    std::partial_sum(num_selected_keys_per_bucket_offset_[gpu_id].begin(),
                     num_selected_keys_per_bucket_offset_[gpu_id].end(),
                     num_selected_keys_per_bucket_offset_[gpu_id].begin());

    int batch_size_per_gpu = batch_size / num_gpus_;
    auto &model_comm_buffer = model_comm_buffers_[gpu_id];
    for (int i = 0; i < metas_[gpu_id].num_network_dst_look; ++i) {
      int lookup_id = metas_[gpu_id].network_dst_lookup_ids[i];

      for (int b = 0; b < batch_size_per_gpu; ++b) {
        int bucket_id = batch_size * lookup_id + gpu_id * batch_size_per_gpu + b;
        int start = static_cast<int>(bucket_range[bucket_id]);
        int end = static_cast<int>(bucket_range[bucket_id + 1]);
        pooling_factor_per_bucket_[gpu_id].push_back(end - start);
      }
    }

    for (int ggpu_id = 0; ggpu_id < num_gpus_; ++ggpu_id) {
      for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
        int table_id = metas_[gpu_id].table_ids[i];
        int ev_size = metas_[gpu_id].ev_sizes[i];
        int ev_offset = metas_[gpu_id].ev_offsets[i];

        for (int b = 0; b < batch_size_per_gpu; ++b) {
          int i_bucket = batch_size * i + batch_size_per_gpu * ggpu_id + b;
          uint32_t start = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket];
          uint32_t end = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket + 1];
          ArrayView<emb_t> model_comm_ev{
              &model_comm_buffer[ggpu_id][batch_size_per_gpu * ev_offset + b * ev_size], ev_size};

          std::vector<float> accumulate_vec;
          accumulate_vec.assign(ev_size, 0.f);
          for (uint32_t r = 0; r < (end - start); ++r) {
            key_t k = selected_keys_[gpu_id][start + r];
            ASSERT_TRUE(table_id < static_cast<int>(emb_table_cpu.emb_table_list_.size()));
            ASSERT_TRUE(emb_table_cpu.emb_table_list_[table_id].find(k) !=
                        emb_table_cpu.emb_table_list_[table_id].end());
            auto ev = emb_table_cpu.emb_table_list_[table_id].at(k);
            ASSERT_TRUE(ev.size() == accumulate_vec.size());

            for (int e = 0; e < ev_size; ++e) {
              accumulate_vec[e] += ev[e];
            }
          }

          for (int e = 0; e < ev_size; ++e) {
            model_comm_ev[e] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[e]);
          }
        }
      }
    }
  }

  void network_forward_per_gpu(int gpu_id, int batch_size, std::vector<emb_t> &embedding_vec) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    auto &network_comm_buffer = network_comm_buffers_[gpu_id];

    for (int i = 0; i < metas_[gpu_id].num_network_dst_look; ++i) {
      int dst_lookup_id = metas_[gpu_id].network_dst_lookup_ids[i];
      Combiner dst_combiner = metas_[gpu_id].network_dst_combiners[i];

      int dst_buffer_ev_offset = metas_[gpu_id].dst_buffer_ev_offsets[dst_lookup_id];
      int dst_buffer_ev_size = metas_[gpu_id].dst_buffer_ev_sizes[dst_lookup_id];
      int start = metas_[gpu_id].network_offsets[i];
      int end = metas_[gpu_id].network_offsets[i + 1];
      for (int b = 0; b < batch_size_per_gpu; ++b) {
        ArrayView<emb_t> dst_buffer_ev{
            &embedding_vec[dst_buffer_ev_offset * batch_size_per_gpu + b * dst_buffer_ev_size],
            dst_buffer_ev_size};
        std::vector<float> accumulate_vec;
        accumulate_vec.assign(dst_buffer_ev_size, 0.f);

        for (int r = start; r < end; ++r) {
          int network_gpu_id = metas_[gpu_id].network_gpu_ids[r];
          int network_id = metas_[gpu_id].network_ids[r];
          int ev_offset = metas_[gpu_id].network_ev_offsets[network_gpu_id][network_id];
          int ev_size = metas_[gpu_id].network_ev_sizes[network_gpu_id][network_id];
          ASSERT_EQ(ev_size, dst_buffer_ev_size);

          ArrayView<emb_t> network_ev{
              &network_comm_buffer[network_gpu_id][batch_size_per_gpu * ev_offset + b * ev_size],
              ev_size};

          for (int e = 0; e < ev_size; ++e) {
            accumulate_vec[e] += HugeCTR::TypeConvert<float, emb_t>::convert(network_ev[e]);
          }
        }

        int dst_bucket_id = batch_size_per_gpu * i + b;
        int pooling_factor = pooling_factor_per_bucket_[gpu_id][dst_bucket_id];
        if (dst_combiner == Combiner::Average && pooling_factor > 0) {
          for (int e = 0; e < dst_buffer_ev_size; ++e) {
            accumulate_vec[e] /= pooling_factor;
          }
        }
        for (int e = 0; e < dst_buffer_ev_size; ++e) {
          dst_buffer_ev[e] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[e]);
        }
      }
    }
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range,
                             const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                             std::vector<std::vector<emb_t>> &embedding_vec, int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      model_forward_per_gpu(gpu_id, keys, bucket_range, emb_table_cpu, batch_size);
    }
    all2all(model_comm_buffers_, network_comm_buffers_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      network_forward_per_gpu(gpu_id, batch_size, embedding_vec[gpu_id]);
    }
  }

  void network_backward_per_gpu(int gpu_id, const std::vector<emb_t> &top_grad, int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    auto &network_comm_buffer = network_comm_buffers_[gpu_id];

    for (int i = 0; i < metas_[gpu_id].num_network_dst_look; ++i) {
      int dst_lookup_id = metas_[gpu_id].network_dst_lookup_ids[i];
      Combiner dst_combiner = metas_[gpu_id].network_dst_combiners[i];

      int dst_buffer_ev_offset = metas_[gpu_id].dst_buffer_ev_offsets[dst_lookup_id];
      int dst_buffer_ev_size = metas_[gpu_id].dst_buffer_ev_sizes[dst_lookup_id];
      int start = metas_[gpu_id].network_offsets[i];
      int end = metas_[gpu_id].network_offsets[i + 1];
      for (int b = 0; b < batch_size_per_gpu; ++b) {
        ArrayView<emb_t> dst_buffer_ev{
            const_cast<emb_t *>(
                &top_grad[dst_buffer_ev_offset * batch_size_per_gpu + b * dst_buffer_ev_size]),
            dst_buffer_ev_size};
        std::vector<float> float_grad_ev;
        for (int e = 0; e < dst_buffer_ev_size; ++e) {
          float_grad_ev.push_back(HugeCTR::TypeConvert<float, emb_t>::convert(dst_buffer_ev[e]));
        }

        int dst_bucket_id = batch_size_per_gpu * i + b;
        int pooling_factor = pooling_factor_per_bucket_[gpu_id][dst_bucket_id];
        if (dst_combiner == Combiner::Average && pooling_factor > 0) {
          for (int e = 0; e < dst_buffer_ev_size; ++e) {
            float_grad_ev[e] /= pooling_factor;
          }
        }

        for (int r = start; r < end; ++r) {
          int network_gpu_id = metas_[gpu_id].network_gpu_ids[r];
          int network_id = metas_[gpu_id].network_ids[r];
          int ev_offset = metas_[gpu_id].network_ev_offsets[network_gpu_id][network_id];
          int ev_size = metas_[gpu_id].network_ev_sizes[network_gpu_id][network_id];
          ASSERT_EQ(ev_size, dst_buffer_ev_size);

          ArrayView<emb_t> network_ev{
              &network_comm_buffer[network_gpu_id][batch_size_per_gpu * ev_offset + b * ev_size],
              ev_size};

          for (int e = 0; e < ev_size; ++e) {
            network_ev[e] = HugeCTR::TypeConvert<emb_t, float>::convert(float_grad_ev[e]);
          }
        }
      }
    }
  }

  void model_backward_per_gpu(
      int gpu_id, int batch_size,
      std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info) {
    int batch_size_per_gpu = batch_size / num_gpus_;
    auto &model_comm_buffer = model_comm_buffers_[gpu_id];

    std::vector<std::unordered_map<key_t, std::vector<std::vector<float>>>> local_reduce_grad;
    local_reduce_grad.resize(grad_info.size());
    for (int ggpu_id = 0; ggpu_id < num_gpus_; ++ggpu_id) {
      for (int i = 0; i < metas_[gpu_id].num_lookup; ++i) {
        int table_id = metas_[gpu_id].table_ids[i];
        ASSERT_TRUE(table_id < static_cast<int>(grad_info.size()));
        auto &wgrad_dict = local_reduce_grad[table_id];
        int ev_size = metas_[gpu_id].ev_sizes[i];
        int ev_offset = metas_[gpu_id].ev_offsets[i];

        for (int b = 0; b < batch_size_per_gpu; ++b) {
          int i_bucket = batch_size * i + batch_size_per_gpu * ggpu_id + b;
          uint32_t start = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket];
          uint32_t end = num_selected_keys_per_bucket_offset_[gpu_id][i_bucket + 1];
          ArrayView<emb_t> model_comm_ev{
              &model_comm_buffer[ggpu_id][batch_size_per_gpu * ev_offset + b * ev_size], ev_size};

          for (uint32_t r = 0; r < (end - start); ++r) {
            key_t k = selected_keys_[gpu_id][start + r];
            std::vector<float> evs;
            for (int e = 0; e < model_comm_ev.size(); ++e) {
              evs.push_back(HugeCTR::TypeConvert<float, emb_t>::convert(model_comm_ev[e]));
            }
            ASSERT_EQ(evs.size(), ev_size);
            wgrad_dict[k].push_back(evs);
          }
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
      network_backward_per_gpu(gpu_id, top_grads[gpu_id], batch_size);
    }
    all2all(network_comm_buffers_, model_comm_buffers_);

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      model_backward_per_gpu(gpu_id, batch_size, grad_info);
    }
  }
};