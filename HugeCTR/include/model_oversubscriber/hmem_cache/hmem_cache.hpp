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

#include <model_oversubscriber/hmem_cache/sparse_model_file_ts.hpp>

namespace {

std::unordered_map<HugeCTR::Optimizer_t, size_t> vec_per_line = {
    {HugeCTR::Optimizer_t::Adam, 3},
    {HugeCTR::Optimizer_t::AdaGrad, 2},
    {HugeCTR::Optimizer_t::MomentumSGD, 2},
    {HugeCTR::Optimizer_t::Nesterov, 2},
    {HugeCTR::Optimizer_t::SGD, 1}};

}

namespace HugeCTR {

struct HMemCacheConfig {
  size_t num_cached_pass;
  double target_hit_rate;
  size_t max_num_evict;
  size_t block_capacity;
  HMemCacheConfig() {}
  HMemCacheConfig(size_t _num_cached_pass, double _target_hit_rate, size_t _max_num_evict)
      : num_cached_pass(_num_cached_pass),
        target_hit_rate(_target_hit_rate),
        max_num_evict(_max_num_evict) {}
};

template <typename TypeKey>
class HMemCache {
 public:
  using HashTableType = typename SparseModelFileTS<TypeKey>::HashTableType;
  size_t static const end_flag{SparseModelFileTS<TypeKey>::end_flag};

 private:
  int const num_block_;
  double const target_hit_rate_;
  size_t const max_num_evict_;
  size_t const block_capacity_;

  const bool use_slot_id_;
  const size_t emb_vec_size_;
  const size_t vec_per_line_;
  std::shared_ptr<ResourceManager> resource_manager_;

  std::vector<HashTableType> key_idx_maps_;
  std::vector<std::vector<size_t>> slot_ids_;
  std::vector<std::vector<std::vector<float>>> cache_datas_;

  bool is_full_{false};
  int head_id_{-1};

  std::shared_ptr<SparseModelFileTS<TypeKey>> sparse_model_file_ptr_;

  size_t find_(TypeKey key);
  std::pair<int, size_t> cascade_find_(TypeKey key);

 public:
  HMemCache(size_t num_cached_pass, double target_hit_rate, size_t max_num_evict,
            size_t max_vocabulary_size, std::string sparse_model_file, std::string local_path,
            bool use_slot_id, Optimizer_t opt_type, size_t emb_vec_size,
            std::shared_ptr<ResourceManager> resource_manager);

  std::pair<std::vector<long long>, std::vector<float>> read(long long const *key_ptr, size_t len);
  void read(TypeKey *key_ptr, size_t &len, size_t *slot_id_ptr, std::vector<float *> &data_ptrs);
  void write(const TypeKey *key_ptr, size_t len, size_t const *slot_id_ptr,
             std::vector<float *> &data_ptrs);

  void sync_to_ssd();

  auto get_sparse_model_file() { return sparse_model_file_ptr_; }
};

}  // namespace HugeCTR
