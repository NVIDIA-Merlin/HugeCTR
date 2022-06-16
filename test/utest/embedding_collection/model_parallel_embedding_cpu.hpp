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
#include "embedding_table_cpu.hpp"
#include "view_cpu.hpp"
using namespace embedding;

template <typename emb_t>
struct RaggedModelBufferCPU {
  std::vector<std::vector<emb_t>> data_;

  std::vector<int> local_ev_offset_list_;
  int num_gpus_;
  int num_embedding_;
  int batch_size_;

  RaggedModelBufferCPU(int num_gpus, int batch_size, const std::vector<int> &local_ev_size_list)
      : local_ev_offset_list_{0},
        num_gpus_(num_gpus),
        num_embedding_(local_ev_size_list.size()),
        batch_size_(batch_size) {
    assert(batch_size % num_gpus == 0);
    for (int i : local_ev_size_list) {
      local_ev_offset_list_.push_back(i);
    }
    std::partial_sum(local_ev_offset_list_.begin(), local_ev_offset_list_.end(),
                     local_ev_offset_list_.begin());

    data_.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      data_[gpu_id].resize(batch_size_ * local_ev_offset_list_.back() / num_gpus_);
    }
  }
};

template <typename emb_t>
struct RaggedNetworkBufferCPU {
  std::vector<std::vector<emb_t>> data_;
  std::vector<std::vector<int>> global_embedding_list_;
  std::vector<int> ev_size_list_;
  int num_gpus_;
  int num_embedding_;
  int batch_size_;
  int batch_size_per_gpu_;

  std::vector<int> gpu_idx_offset_;
  std::vector<std::vector<int>> global_ev_offset_;

  RaggedNetworkBufferCPU(int batch_size, const std::vector<std::vector<int>> &global_embedding_list,
                         const std::vector<int> &ev_size_list)
      : global_embedding_list_(global_embedding_list),
        ev_size_list_(ev_size_list),
        num_gpus_(global_embedding_list.size()),
        num_embedding_(0),
        batch_size_(batch_size),
        batch_size_per_gpu_(batch_size / num_gpus_),
        gpu_idx_offset_{0} {
    assert(batch_size % num_gpus_ == 0);

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      num_embedding_ += global_embedding_list_[gpu_id].size();
    }

    data_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      auto &local_embedding_list = global_embedding_list_[gpu_id];

      int num_elems = 0;
      for (int embedding_id : local_embedding_list) {
        num_elems += ev_size_list_[embedding_id];
      }
      data_[gpu_id].resize(num_elems * batch_size_per_gpu_);

      std::vector<int> local_ev_offset{0};
      for (int embedding_id : local_embedding_list) {
        local_ev_offset.push_back(ev_size_list_[embedding_id]);
      }
      global_ev_offset_.push_back(local_ev_offset);

      int num_local_embedding = local_embedding_list.size();
      gpu_idx_offset_.push_back(num_local_embedding * batch_size_per_gpu_);
    }

    std::partial_sum(gpu_idx_offset_.begin(), gpu_idx_offset_.end(), gpu_idx_offset_.begin());

    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      std::partial_sum(global_ev_offset_[gpu_id].begin(), global_ev_offset_[gpu_id].end(),
                       global_ev_offset_[gpu_id].begin());
    }
  }
};

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class ModelParallelEmbeddingCPU {
  int num_gpus_;
  int num_embedding_;
  int num_table_;
  EmbeddingCollectionParam ebc_param_;
  std::vector<EmbeddingShardingParam> sharding_param_list_;

  std::vector<int> ev_size_list_;
  std::vector<int> ev_offset_list_;
  std::vector<char> combiner_list_;

  std::vector<std::vector<int>> local_id_space_list_;
  std::vector<std::vector<int>> local_embedding_list_;
  std::vector<std::vector<int>> local_hotness_list_;
  std::vector<std::vector<int>> local_ev_size_list_;
  std::vector<std::vector<int>> local_ev_offset_list_;
  std::vector<std::vector<int>> global_embedding_list_;
  std::vector<int> sharding_id_list_;
  std::vector<int> num_sharding_list_;

  std::vector<std::vector<key_t>> mp_model_key_list_;
  std::vector<std::vector<uint32_t>> mp_model_offset_list_;
  std::vector<std::vector<int>> mp_model_dst_list_;
  std::vector<RaggedModelBufferCPU<emb_t>> model_buffer_list_;

  std::vector<std::vector<int>> network_idx_list_;
  std::vector<std::vector<int>> network_offset_list_;
  std::vector<std::vector<int>> network_dst_list_;
  std::vector<RaggedNetworkBufferCPU<emb_t>> network_buffer_list_;

  std::vector<std::vector<key_t>> unique_key_list_;
  std::vector<std::vector<uint32_t>> sorted_bucket_id_list_;
  std::vector<std::vector<uint32_t>> sorted_bucket_id_offset_list_;
  std::vector<std::vector<int>> num_unique_key_scan_list_;

  std::vector<RaggedGradBufferCPU<float>> grad_;

 public:
  ModelParallelEmbeddingCPU(int num_gpus, int num_table, const EmbeddingCollectionParam &ebc_param,
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
      combiner_list_.push_back(static_cast<char>(ebc_param_.embedding_params[i].combiner));
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());

    local_id_space_list_.resize(num_gpus_);
    local_embedding_list_.resize(num_gpus_);
    local_hotness_list_.resize(num_gpus_);
    local_ev_size_list_.resize(num_gpus_);
    local_ev_offset_list_.resize(num_gpus_);

    auto &embedding_params = ebc_param_.embedding_params;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      local_ev_offset_list_[gpu_id].push_back(0);
      for (int embedding_id : sharding_param_list[gpu_id].local_embedding_list) {
        local_id_space_list_[gpu_id].push_back(embedding_params[embedding_id].id_space);
        local_embedding_list_[gpu_id].push_back(embedding_id);
        local_hotness_list_[gpu_id].push_back(embedding_params[embedding_id].hotness);
        local_ev_size_list_[gpu_id].push_back(embedding_params[embedding_id].ev_size);
        local_ev_offset_list_[gpu_id].push_back(embedding_params[embedding_id].ev_size);
      }
      std::inclusive_scan(local_ev_offset_list_[gpu_id].begin(),
                          local_ev_offset_list_[gpu_id].end(),
                          local_ev_offset_list_[gpu_id].begin());
      global_embedding_list_.push_back(sharding_param_list[gpu_id].local_embedding_list);
      sharding_id_list_.push_back(sharding_param_list[gpu_id].sharding_id);
      num_sharding_list_.push_back(sharding_param_list[gpu_id].num_sharding);
    }

    mp_model_key_list_.resize(num_gpus);
    mp_model_offset_list_.resize(num_gpus);
    mp_model_dst_list_.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      model_buffer_list_.emplace_back(num_gpus, ebc_param.universal_batch_size,
                                      local_ev_size_list_[gpu_id]);
    }

    network_idx_list_.resize(num_gpus);
    network_offset_list_.resize(num_gpus);
    network_dst_list_.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      network_buffer_list_.emplace_back(ebc_param.universal_batch_size, global_embedding_list_,
                                        ev_size_list_);
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

  void all2all(const std::vector<std::vector<std::vector<emb_t>> *> &send_buffer,
               std::vector<std::vector<std::vector<emb_t>> *> &recv_buffer) {
    for (int src_gpu_id = 0; src_gpu_id < num_gpus_; ++src_gpu_id) {
      for (int dst_gpu_id = 0; dst_gpu_id < num_gpus_; ++dst_gpu_id) {
        auto &send_tensor = (*send_buffer[src_gpu_id])[dst_gpu_id];
        auto &recv_tensor = (*recv_buffer[dst_gpu_id])[src_gpu_id];
        recv_tensor = send_tensor;
      }
    }
  }

  void cpu_mp_model_index_calculation_per_gpu(int gpu_id, const std::vector<key_t> &keys,
                                              const std::vector<offset_t> &bucket_range,
                                              int batch_size) {
    auto local_embedding_list = local_embedding_list_[gpu_id];
    int sharding_id = sharding_id_list_[gpu_id];
    int num_sharding = num_sharding_list_[gpu_id];

    mp_model_key_list_[gpu_id].clear();
    mp_model_offset_list_[gpu_id].clear();
    mp_model_offset_list_[gpu_id].push_back(0);
    mp_model_dst_list_[gpu_id].clear();
    for (int bucket_id = 0; bucket_id < static_cast<int>(local_embedding_list.size()) * batch_size;
         ++bucket_id) {
      int local_embedding_id = bucket_id / batch_size;
      int embedding_id = local_embedding_list[local_embedding_id];
      int batch_id = bucket_id % batch_size;

      int global_bucket_id = batch_size * embedding_id + batch_id;
      offset_t start = bucket_range[global_bucket_id];
      offset_t end = bucket_range[global_bucket_id + 1];
      uint32_t num_key_in_bucket = 0;
      for (offset_t i = 0; i < (end - start); ++i) {
        key_t k = keys[start + i];
        if (static_cast<int>(k) % num_sharding == sharding_id) {
          mp_model_key_list_[gpu_id].push_back(keys[start + i]);
          ++num_key_in_bucket;
        }
      }

      mp_model_offset_list_[gpu_id].push_back(num_key_in_bucket);
      mp_model_dst_list_[gpu_id].push_back(bucket_id);
    }
    std::inclusive_scan(mp_model_offset_list_[gpu_id].begin(), mp_model_offset_list_[gpu_id].end(),
                        mp_model_offset_list_[gpu_id].begin());
  }

  void cpu_mp_model_forward_per_gpu(int gpu_id,
                                    const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                                    int batch_size) {
    auto local_id_space_list = local_id_space_list_[gpu_id];
    auto local_embedding_list = local_embedding_list_[gpu_id];

    RaggedModelBufferViewCPU<emb_t> model_view{&model_buffer_list_[gpu_id].data_,
                                               &model_buffer_list_[gpu_id].local_ev_offset_list_,
                                               num_gpus_, batch_size};
    assert(num_gpus_ == model_buffer_list_[gpu_id].num_gpus_);
    assert(batch_size == model_buffer_list_[gpu_id].batch_size_);

    for (int bucket_id = 0; bucket_id < static_cast<int>(local_embedding_list.size()) * batch_size;
         ++bucket_id) {
      int local_emb_id = bucket_id / batch_size;

      int id_space = local_id_space_list[local_emb_id];

      int dst_bucket_id = mp_model_dst_list_[gpu_id][bucket_id];
      ArrayView<emb_t> dst_ev = model_view[dst_bucket_id];
      std::vector<float> accumulate_vec;
      accumulate_vec.assign(dst_ev.size(), 0.f);

      uint32_t start = mp_model_offset_list_[gpu_id][bucket_id];
      uint32_t end = mp_model_offset_list_[gpu_id][bucket_id + 1];

      for (uint32_t r = 0; r < (end - start); ++r) {
        key_t k = mp_model_key_list_[gpu_id][start + r];
        ASSERT_TRUE(emb_table_cpu.emb_table_list_[id_space].find(k) !=
                    emb_table_cpu.emb_table_list_[id_space].end());
        auto ev = emb_table_cpu.emb_table_list_[id_space].at(k);
        assert(ev.size() == accumulate_vec.size());

        for (int i = 0; i < dst_ev.size(); ++i) {
          accumulate_vec[i] += ev[i];
        }
      }

      for (int i = 0; i < dst_ev.size(); ++i) {
        dst_ev[i] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[i]);
      }
    }
  }

  void cpu_mp_network_index_calculation_per_gpu(int gpu_id) {
    int batch_size = ebc_param_.universal_batch_size;
    int batch_size_per_gpu = batch_size / num_gpus_;

    network_idx_list_[gpu_id].clear();
    network_offset_list_[gpu_id].clear();
    network_offset_list_[gpu_id].push_back(0);
    network_dst_list_[gpu_id].clear();

    std::vector<int> dst_embedding_id_list;
    for (auto &vec : global_embedding_list_) {
      dst_embedding_id_list.insert(dst_embedding_id_list.end(), vec.begin(), vec.end());
    }

    std::sort(dst_embedding_id_list.begin(), dst_embedding_id_list.end());
    auto last = std::unique(dst_embedding_id_list.begin(), dst_embedding_id_list.end());
    dst_embedding_id_list.erase(last, dst_embedding_id_list.end());
    for (int dst_embedding_id : dst_embedding_id_list) {
      for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
        network_dst_list_[gpu_id].push_back(batch_size_per_gpu * dst_embedding_id + batch_id);
      }
    }

    std::vector<int> num_embedding_offset{0};
    for (auto &vec : global_embedding_list_) {
      num_embedding_offset.push_back(vec.size());
    }
    std::partial_sum(num_embedding_offset.begin(), num_embedding_offset.end(),
                     num_embedding_offset.begin());

    std::vector<int> network_embedding_list;
    std::vector<int> network_embedding_offset{0};

    network_embedding_offset.assign(dst_embedding_id_list.size() + 1, 0);

    for (int local_embedding_id = 0;
         local_embedding_id < static_cast<int>(dst_embedding_id_list.size());
         ++local_embedding_id) {
      int dst_embedding_id = dst_embedding_id_list[local_embedding_id];

      for (int src_gpu_id = 0; src_gpu_id < num_gpus_; ++src_gpu_id) {
        auto iter = std::find(global_embedding_list_[src_gpu_id].begin(),
                              global_embedding_list_[src_gpu_id].end(), dst_embedding_id);
        if (iter == global_embedding_list_[src_gpu_id].end()) continue;
        int idx = std::distance(global_embedding_list_[src_gpu_id].begin(), iter);

        network_embedding_list.push_back(num_embedding_offset[src_gpu_id] + idx);
        network_embedding_offset[1 + local_embedding_id] += 1;
      }
    }
    std::inclusive_scan(network_embedding_offset.begin(), network_embedding_offset.end(),
                        network_embedding_offset.begin());

    for (size_t i = 0; i < dst_embedding_id_list.size(); ++i) {
      int start = network_embedding_offset[i];
      int end = network_embedding_offset[i + 1];

      for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
        for (int r = start; r < end; ++r) {
          int embedding_id = network_embedding_list[r];
          network_idx_list_[gpu_id].push_back(embedding_id * batch_size_per_gpu + batch_id);
        }
        network_offset_list_[gpu_id].push_back(end - start);
      }
    }

    std::inclusive_scan(network_offset_list_[gpu_id].begin(), network_offset_list_[gpu_id].end(),
                        network_offset_list_[gpu_id].begin());
  }

  void cpu_mp_network_forward_per_gpu(int gpu_id, int batch_size,
                                      const std::vector<offset_t> &bucket_range,
                                      std::vector<std::vector<emb_t>> &embedding_vec) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    assert(batch_size == network_buffer_list_[gpu_id].batch_size_);
    assert(batch_size_per_gpu == network_buffer_list_[gpu_id].batch_size_per_gpu_);

    RaggedNetworkBufferViewCPU<emb_t> network_view{
        &network_buffer_list_[gpu_id].data_, &network_buffer_list_[gpu_id].gpu_idx_offset_,
        &network_buffer_list_[gpu_id].global_ev_offset_, network_buffer_list_[gpu_id].num_gpus_,
        network_buffer_list_[gpu_id].batch_size_};

    RaggedEmbForwardResultViewCPU<emb_t> result_view{&embedding_vec[gpu_id], &ev_size_list_,
                                                     &ev_offset_list_, batch_size_per_gpu};

    for (size_t idx = 0; idx < network_dst_list_[gpu_id].size(); ++idx) {
      int start = network_offset_list_[gpu_id][idx];
      int end = network_offset_list_[gpu_id][idx + 1];
      int dst_idx = network_dst_list_[gpu_id][idx];
      int dst_embeding_id = dst_idx / batch_size_per_gpu;
      int dst_batch_id = dst_idx % batch_size_per_gpu;
      int combiner = combiner_list_[dst_embeding_id];
      ArrayView<emb_t> dst_tensor = result_view[dst_idx];

      std::vector<float> accumulate_vec;
      accumulate_vec.assign(dst_tensor.size(), 0.f);
      for (int r = start; r < end; ++r) {
        int src_idx = network_idx_list_[gpu_id][r];
        ArrayView<emb_t> src_tensor = network_view[src_idx];

        assert(src_tensor.size() == dst_tensor.size());
        for (int i = 0; i < dst_tensor.size(); ++i) {
          accumulate_vec[i] += HugeCTR::TypeConvert<float, emb_t>::convert(src_tensor[i]);
        }
      }

      int dst_bucket_id = batch_size * dst_embeding_id + batch_size_per_gpu * gpu_id + dst_batch_id;
      int num_key_in_bucket = bucket_range[dst_bucket_id + 1] - bucket_range[dst_bucket_id];
      if (combiner == static_cast<char>(Combiner::Average) && num_key_in_bucket > 0) {
        for (int i = 0; i < dst_tensor.size(); ++i) {
          accumulate_vec[i] /= num_key_in_bucket;
        }
      }

      for (int i = 0; i < dst_tensor.size(); ++i) {
        dst_tensor[i] = HugeCTR::TypeConvert<emb_t, float>::convert(accumulate_vec[i]);
      }
    }
  }

  void cpu_mp_network_backward_per_gpu(int gpu_id, const std::vector<std::vector<emb_t>> &top_grads,
                                       int batch_size) {
    int batch_size_per_gpu = batch_size / num_gpus_;

    assert(batch_size == network_buffer_list_[gpu_id].batch_size_);
    assert(batch_size_per_gpu == network_buffer_list_[gpu_id].batch_size_per_gpu_);

    RaggedEmbForwardResultViewCPU<emb_t> grad_view{
        const_cast<std::vector<emb_t> *>(&top_grads[gpu_id]), &ev_size_list_, &ev_offset_list_,
        batch_size_per_gpu};

    RaggedNetworkBufferViewCPU<emb_t> network_view{
        &network_buffer_list_[gpu_id].data_, &network_buffer_list_[gpu_id].gpu_idx_offset_,
        &network_buffer_list_[gpu_id].global_ev_offset_, network_buffer_list_[gpu_id].num_gpus_,
        network_buffer_list_[gpu_id].batch_size_};

    for (int idx = 0; idx < static_cast<int>(network_offset_list_[gpu_id].size()) - 1; ++idx) {
      int start = network_offset_list_[gpu_id][idx];
      int end = network_offset_list_[gpu_id][idx + 1];
      int dst_idx = network_dst_list_[gpu_id][idx];

      ArrayView<emb_t> grad = grad_view[dst_idx];

      for (int r = start; r < end; ++r) {
        int src_idx = network_idx_list_[gpu_id][r];
        ArrayView<emb_t> dst_tensor = network_view[src_idx];
        assert(grad.size() == dst_tensor.size());
        for (int i = 0; i < dst_tensor.size(); ++i) {
          dst_tensor[i] = grad[i];
        }
      }
    }
  }

  void cpu_mp_model_backward_index_calculation_per_gpu(int gpu_id, int batch_size) {
    unique_key_list_[gpu_id].clear();
    sorted_bucket_id_list_[gpu_id].clear();
    sorted_bucket_id_offset_list_[gpu_id].clear();
    sorted_bucket_id_offset_list_[gpu_id].push_back(0);
    num_unique_key_scan_list_[gpu_id].clear();
    num_unique_key_scan_list_[gpu_id].push_back(0);

    std::vector<int> local_id_space_list = local_id_space_list_[gpu_id];
    auto &unique_id_space_list = grad_[gpu_id].unique_id_space_list_;

    std::vector<std::unordered_map<key_t, std::vector<uint32_t>>> tmp_backward_info;
    tmp_backward_info.resize(unique_id_space_list.size());
    // std::cout << "collecting backward:\n";
    for (size_t local_embedding_id = 0; local_embedding_id < local_embedding_list_[gpu_id].size();
         ++local_embedding_id) {
      int id_space = local_id_space_list[local_embedding_id];
      auto find_unique_id_space =
          std::find(unique_id_space_list.begin(), unique_id_space_list.end(), id_space);
      ASSERT_TRUE(find_unique_id_space != unique_id_space_list.end());
      int unique_id_space_idx = find_unique_id_space - unique_id_space_list.begin();

      // std::cout << "id_space:" << id_space << "\n";
      for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        int bucket_id = local_embedding_id * batch_size + batch_id;
        int dst_bucket_id = mp_model_dst_list_[gpu_id][bucket_id];

        int start = mp_model_offset_list_[gpu_id][bucket_id];
        int end = mp_model_offset_list_[gpu_id][bucket_id + 1];
        for (int i = start; i < end; ++i) {
          key_t k = mp_model_key_list_[gpu_id][i];
          // std::cout << "k:" << k << ",bucket_id:" << dst_bucket_id << "\n";
          tmp_backward_info[unique_id_space_idx][k].push_back(dst_bucket_id);
        }
      }
    }
    // std::cout << "mp backward info:\n";
    // for (size_t i = 0; i < unique_id_space_list.size(); ++i) {
    //   int id_space = unique_id_space_list[i];
    //   std::cout << "id_space:" << id_space << ",key:";
    //   for (auto &[key, bucket_id_list] : tmp_backward_info[i]) {
    //     std::cout << key << " ";
    //   }
    //   std::cout << "\n";
    // }

    for (auto &backward_info_in_one_id_space : tmp_backward_info) {
      for (auto &p : backward_info_in_one_id_space) {
        unique_key_list_[gpu_id].push_back(p.first);
        sorted_bucket_id_list_[gpu_id].insert(sorted_bucket_id_list_[gpu_id].end(),
                                              p.second.begin(), p.second.end());
        sorted_bucket_id_offset_list_[gpu_id].push_back(p.second.size());
      }
      num_unique_key_scan_list_[gpu_id].push_back(backward_info_in_one_id_space.size());
    }
    std::partial_sum(num_unique_key_scan_list_[gpu_id].begin(),
                     num_unique_key_scan_list_[gpu_id].end(),
                     num_unique_key_scan_list_[gpu_id].begin());
    std::partial_sum(sorted_bucket_id_offset_list_[gpu_id].begin(),
                     sorted_bucket_id_offset_list_[gpu_id].end(),
                     sorted_bucket_id_offset_list_[gpu_id].begin());
  }

  void cpu_mp_model_backward_per_gpu(
      int gpu_id, int batch_size,
      std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info) {
    RaggedModelBufferViewCPU<emb_t> model_view{&model_buffer_list_[gpu_id].data_,
                                               &model_buffer_list_[gpu_id].local_ev_offset_list_,
                                               num_gpus_, batch_size};
    assert(num_gpus_ == model_buffer_list_[gpu_id].num_gpus_);
    assert(batch_size == model_buffer_list_[gpu_id].batch_size_);

    // std::cout << "gpu_id:" << gpu_id << ",cpu_mp_model_backward_per_gpu model_view:\n";
    // for (auto &j : model_buffer_list_[gpu_id].data_) {
    //   for(emb_t i: j) {
    //     std::cout << HugeCTR::TypeConvert<float, emb_t>::convert(i) << " ";
    //   }
    // }
    // std::cout << "\n";

    std::vector<int> ev_size_scan_list{0};
    for (size_t idx = 0; idx < grad_[gpu_id].unique_ev_size_list_.size(); ++idx) {
      int start = num_unique_key_scan_list_[gpu_id][idx];
      int end = num_unique_key_scan_list_[gpu_id][idx + 1];
      for (int i = start; i < end; ++i) {
        ev_size_scan_list.push_back(grad_[gpu_id].unique_ev_size_list_[idx]);
      }
    }
    std::partial_sum(ev_size_scan_list.begin(), ev_size_scan_list.end(), ev_size_scan_list.begin());
    assert(grad_[gpu_id].grad_.size() >= ev_size_scan_list.back());

    RaggedGradBufferViewCPU<float> grad_view{&ev_size_scan_list, &grad_[gpu_id].grad_};

    for (int idx = 0; idx < num_unique_key_scan_list_[gpu_id].back(); ++idx) {
      int start = sorted_bucket_id_offset_list_[gpu_id][idx];
      int end = sorted_bucket_id_offset_list_[gpu_id][idx + 1];

      ArrayView<float> dst_ev = grad_view[idx];
      std::vector<float> accumulate_vec;
      accumulate_vec.assign(dst_ev.size(), 0.f);

      for (int r = start; r < end; ++r) {
        int bucket_id = sorted_bucket_id_list_[gpu_id][r];
        auto src_ev = model_view[bucket_id];
        assert(src_ev.size() == dst_ev.size());

        for (int i = 0; i < dst_ev.size(); ++i) {
          accumulate_vec[i] += HugeCTR::TypeConvert<float, emb_t>::convert(src_ev[i]);
        }
      }

      for (int i = 0; i < dst_ev.size(); ++i) {
        dst_ev[i] = accumulate_vec[i];
      }
    }

    for (size_t idx = 0; idx < grad_[gpu_id].unique_id_space_list_.size(); ++idx) {
      int id_space = grad_[gpu_id].unique_id_space_list_[idx];

      int start = num_unique_key_scan_list_[gpu_id][idx];
      int end = num_unique_key_scan_list_[gpu_id][idx + 1];

      for (int r = start; r < end; ++r) {
        key_t k = unique_key_list_[gpu_id][r];
        auto ev = grad_view[r];
        if (grad_info[id_space].find(k) == grad_info[id_space].end()) {
          grad_info[id_space][k].assign(ev.size(), 0.f);
        }
        for (int i = 0; i < ev.size(); ++i) {
          grad_info[id_space][k][i] = ev[i];
        }
      }
    }
  }

  void embedding_forward_cpu(const std::vector<key_t> &t_keys,
                             const std::vector<offset_t> &flatten_concat_bucket_range,
                             const EmbeddingTableCPU<key_t, index_t> &emb_table_cpu,
                             std::vector<std::vector<emb_t>> &embedding_vec, int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_mp_model_index_calculation_per_gpu(gpu_id, t_keys, flatten_concat_bucket_range,
                                             batch_size);
      cpu_mp_model_forward_per_gpu(gpu_id, emb_table_cpu, batch_size);
      cpu_mp_network_index_calculation_per_gpu(gpu_id);
      cpu_mp_model_backward_index_calculation_per_gpu(gpu_id, batch_size);
    }
    std::vector<std::vector<std::vector<emb_t>> *> model_buffer_ptr;
    for (auto &buffer : model_buffer_list_) {
      model_buffer_ptr.push_back(&buffer.data_);
    }
    std::vector<std::vector<std::vector<emb_t>> *> network_buffer_ptr;
    for (auto &buffer : network_buffer_list_) {
      network_buffer_ptr.push_back(&buffer.data_);
    }
    all2all(model_buffer_ptr, network_buffer_ptr);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_mp_network_forward_per_gpu(gpu_id, batch_size, flatten_concat_bucket_range,
                                     embedding_vec);
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grads,
                              std::vector<std::unordered_map<key_t, std::vector<float>>> &grad_info,
                              int batch_size) {
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_mp_network_backward_per_gpu(gpu_id, top_grads, batch_size);
    }
    std::vector<std::vector<std::vector<emb_t>> *> model_buffer_ptr;
    for (auto &buffer : model_buffer_list_) {
      model_buffer_ptr.push_back(&buffer.data_);
    }
    std::vector<std::vector<std::vector<emb_t>> *> network_buffer_ptr;
    for (auto &buffer : network_buffer_list_) {
      network_buffer_ptr.push_back(&buffer.data_);
    }
    all2all(network_buffer_ptr, model_buffer_ptr);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      cpu_mp_model_backward_per_gpu(gpu_id, batch_size, grad_info);
    }
  }
};