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

#include <omp.h>
#include <tqdm.h>

#include <cstddef>
#include <embedding_training_cache/hmem_cache/hmem_cache.hpp>
#include <execution>
#include <iomanip>

namespace HugeCTR {

template <typename TypeKey>
size_t HMemCache<TypeKey>::find_(TypeKey key) {
  if (!is_full_ && head_id_ == -1) return end_flag;
  auto num_blk{is_full_ ? num_block_ : (head_id_ + 1)};
  for (int cnt{0}; cnt < num_blk; cnt++) {
    auto blk_idx{(head_id_ + num_block_ - cnt) % num_block_};
    auto it{key_idx_maps_[blk_idx].find(key)};
    if (it != key_idx_maps_[blk_idx].end()) {
      return (blk_idx * block_capacity_ + it->second);
    }
  }
  return end_flag;
}

template <typename TypeKey>
std::pair<int, size_t> HMemCache<TypeKey>::cascade_find_(TypeKey key) {
  auto hc_idx{find_(key)};
  if (hc_idx != end_flag) return std::make_pair(0, hc_idx);
  auto ssd_idx{sparse_model_file_ptr_->find(key)};
  if (ssd_idx != end_flag) return std::make_pair(1, ssd_idx);
  return std::make_pair(-1, 0);
}

template <typename TypeKey>
HMemCache<TypeKey>::HMemCache(size_t num_cached_pass, double target_hit_rate, size_t max_num_evict,
                              size_t max_vocabulary_size, std::string sparse_model_file,
                              std::string local_path, bool use_slot_id, Optimizer_t opt_type,
                              size_t emb_vec_size,
                              std::shared_ptr<ResourceManager> resource_manager)
    : num_block_{static_cast<int>(num_cached_pass)},
      target_hit_rate_{target_hit_rate},
      max_num_evict_{max_num_evict},
      block_capacity_{max_vocabulary_size},
      use_slot_id_{use_slot_id},
      emb_vec_size_{emb_vec_size},
      vec_per_line_{vec_per_line[opt_type]},
      resource_manager_{resource_manager},
      sparse_model_file_ptr_(std::make_shared<SparseModelFileTS<TypeKey>>(
          sparse_model_file, local_path, use_slot_id, opt_type, emb_vec_size, resource_manager)) {
  // +1 is reserved for a temp buffer
  key_idx_maps_.resize(num_block_ + 1);
  slot_ids_.resize(num_block_ + 1);
  cache_datas_.resize(num_block_ + 1);
#pragma omp parallel for num_threads(num_block_ + 1)
  for (auto i = 0; i < num_block_ + 1; i++) {
    key_idx_maps_[i].reserve(block_capacity_);
    if (use_slot_id_) {
      slot_ids_[i].resize(block_capacity_);
    }
    cache_datas_[i].resize(vec_per_line_);
    for (auto &cache_data : cache_datas_[i]) {
      cache_data.resize(block_capacity_ * emb_vec_size_);
    }
  }
}

template <typename TypeKey>
std::pair<std::vector<long long>, std::vector<float>> HMemCache<TypeKey>::read(
    long long const *key_ptr, size_t len) {
  std::vector<TypeKey> keys(len);
  std::transform(std::execution::par, key_ptr, key_ptr + len, keys.begin(),
                 [](long long key) { return static_cast<TypeKey>(key); });
  std::vector<size_t> slot_ids;
  if (use_slot_id_) slot_ids.resize(len);
  std::vector<std::vector<float>> data_vecs(vec_per_line_);
  std::vector<float *> data_ptrs;
  for (auto &data_vec : data_vecs) {
    data_vec.resize(len * emb_vec_size_);
    data_ptrs.push_back(data_vec.data());
  }
  read(keys.data(), len, slot_ids.data(), data_ptrs);
  keys.resize(len);
  if (use_slot_id_) slot_ids.resize(len);
  for (auto &data_vec : data_vecs) {
    data_vec.resize(len * emb_vec_size_);
  }
  std::vector<long long> i64_keys(len);
  std::transform(std::execution::par, keys.begin(), keys.end(), i64_keys.begin(),
                 [](TypeKey key) { return static_cast<long long>(key); });
  return std::pair(std::move(i64_keys), std::move(data_vecs[0]));
}

template <typename TypeKey>
void HMemCache<TypeKey>::read(TypeKey *key_ptr, size_t &len, size_t *slot_id_ptr,
                              std::vector<float *> &data_ptrs) {
  if (data_ptrs.size() != vec_per_line_) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Num of data files and pointers doesn't equal");
  }
  auto const num_thread{24};
  std::vector<std::vector<std::vector<TypeKey>>> sub_exist_keys(
      num_thread, std::vector<std::vector<TypeKey>>(2));
  std::vector<std::vector<std::vector<size_t>>> sub_idx_vecs(num_thread,
                                                             std::vector<std::vector<size_t>>(2));
#pragma omp parallel num_threads(num_thread)
  {
    auto const tid{static_cast<size_t>(omp_get_thread_num())};
    auto sub_chunk_size{len / num_thread};
    auto const idx{sub_chunk_size * tid};
    if (tid == num_thread - 1) sub_chunk_size += (len % num_thread);

    for (size_t i{0}; i < 2; i++) {
      sub_exist_keys[tid][i].reserve(sub_chunk_size);
      sub_idx_vecs[tid][i].reserve(sub_chunk_size);
    }
    for (size_t i{idx}; i < idx + sub_chunk_size; i++) {
      auto key{key_ptr[i]};
      auto idx_pair{cascade_find_(key)};
      if (idx_pair.first == -1) continue;
      sub_exist_keys[tid][idx_pair.first].push_back(key);
      sub_idx_vecs[tid][idx_pair.first].push_back(idx_pair.second);
    }
  }
  std::vector<std::vector<TypeKey>> keys_vec(2);
  std::vector<std::vector<size_t>> idx_vecs(2);
#pragma omp parallel for num_threads(2)
  for (size_t i = 0; i < 2; i++) {
    size_t total_elem{0};
    for (size_t tid{0}; tid < num_thread; tid++) {
      total_elem += sub_exist_keys[tid][i].size();
    }
    if (total_elem == 0) continue;
    keys_vec[i].resize(total_elem);
    idx_vecs[i].resize(total_elem);
    size_t offset{0};
    for (size_t tid{0}; tid < num_thread; tid++) {
      auto num_elem{sub_exist_keys[tid][i].size()};
      if (num_elem == 0) continue;

      auto src_ptr{sub_exist_keys[tid][i].data()};
      auto dst_ptr{keys_vec[i].data() + offset};
      memcpy(dst_ptr, src_ptr, num_elem * sizeof(TypeKey));

      auto src_id_ptr{sub_idx_vecs[tid][i].data()};
      auto dst_id_ptr{idx_vecs[i].data() + offset};
      memcpy(dst_id_ptr, src_id_ptr, num_elem * sizeof(size_t));
      offset += num_elem;
    }
  }
  len = 0;
  for (auto &key_vec : keys_vec) {
    memcpy(key_ptr + len, key_vec.data(), key_vec.size() * sizeof(TypeKey));
    len += key_vec.size();
  }

  size_t static pass_counter{0};
  auto const tail_id{(head_id_ + 1) % num_block_};
  auto hit_rate{(len != 0) ? (1.0 * keys_vec[0].size() / len) : 0.};

#pragma omp parallel for num_threads(24)
  for (size_t cnt = 0; cnt < idx_vecs[0].size(); cnt++) {
    size_t blk_idx{idx_vecs[0][cnt] / block_capacity_};
    size_t line_idx{idx_vecs[0][cnt] % block_capacity_};
    if (use_slot_id_) {
      slot_id_ptr[cnt] = slot_ids_[blk_idx][line_idx];
    }
    for (size_t i{0}; i < vec_per_line_; i++) {
      float *src_ptr{cache_datas_[blk_idx][i].data() + line_idx * emb_vec_size_};
      float *dst_ptr{data_ptrs[i] + cnt * emb_vec_size_};
      memcpy(dst_ptr, src_ptr, emb_vec_size_ * sizeof(float));
    }
  }

  omp_set_nested(2);
#pragma omp parallel sections
  {
#pragma omp section
    {
      if (!is_full_ || (hit_rate < target_hit_rate_ && pass_counter < max_num_evict_)) {
        if (is_full_) {
          std::swap(key_idx_maps_[tail_id], key_idx_maps_[num_block_]);
          std::swap(slot_ids_[tail_id], slot_ids_[num_block_]);
          std::swap(cache_datas_[tail_id], cache_datas_[num_block_]);
        }
        key_idx_maps_[tail_id].clear();
        for (size_t i{0}; i < len; i++) {
          key_idx_maps_[tail_id].insert({key_ptr[i], i});
        }
        bool is_empty{idx_vecs[0].size() == 0};
        if (use_slot_id_ && !is_empty) {
          size_t *src_ptr{slot_id_ptr};
          size_t *dst_ptr{slot_ids_[tail_id].data()};
          memcpy(dst_ptr, src_ptr, idx_vecs[0].size() * sizeof(size_t));
        }
        for (size_t i{0}; (i < vec_per_line_) && !is_empty; i++) {
          float *src_ptr{data_ptrs[i]};
          float *dst_ptr{cache_datas_[tail_id][i].data()};
          memcpy(dst_ptr, src_ptr, idx_vecs[0].size() * emb_vec_size_ * sizeof(float));
        }
      }
    }
#pragma omp section
    {
      size_t *tmp_slot_id_ptr{slot_id_ptr + idx_vecs[0].size()};
      std::vector<float *> tmp_data_ptrs(data_ptrs.size());
      std::transform(data_ptrs.begin(), data_ptrs.end(), tmp_data_ptrs.begin(),
                     [&](float *ptr) { return ptr + idx_vecs[0].size() * emb_vec_size_; });
      sparse_model_file_ptr_->load(idx_vecs[1], tmp_slot_id_ptr, tmp_data_ptrs);
      if (!is_full_ || (hit_rate < target_hit_rate_ && pass_counter < max_num_evict_)) {
        size_t offset{idx_vecs[0].size()};
        bool is_empty{idx_vecs[1].size() == 0};
        if (use_slot_id_ && !is_empty) {
          size_t *src_ptr{tmp_slot_id_ptr};
          size_t *dst_ptr{slot_ids_[tail_id].data() + offset};
          memcpy(dst_ptr, src_ptr, idx_vecs[1].size() * sizeof(size_t));
        }
        for (size_t i{0}; (i < vec_per_line_) && !is_empty; i++) {
          float *src_ptr{tmp_data_ptrs[i]};
          float *dst_ptr{cache_datas_[tail_id][i].data() + offset * emb_vec_size_};
          memcpy(dst_ptr, src_ptr, idx_vecs[1].size() * emb_vec_size_ * sizeof(float));
        }
      }
    }
  }

  if (!is_full_ || (hit_rate < target_hit_rate_ && pass_counter < max_num_evict_)) {
    if (is_full_) {
      sparse_model_file_ptr_->dump_update(key_idx_maps_[num_block_], slot_ids_[num_block_],
                                          cache_datas_[num_block_]);
      pass_counter++;
    }
    head_id_ = tail_id;
    if (!is_full_ && (head_id_ == num_block_ - 1)) {
      is_full_ = true;
    }
  }
  HCTR_LOG_S(INFO, WORLD) << "HMEM-Cache PS: Hit rate [load]: " << std::setprecision(4)
                          << (hit_rate * 100.) << " %" << std::endl;
}

template <typename TypeKey>
void HMemCache<TypeKey>::write(const TypeKey *key_ptr, size_t len, size_t const *slot_id_ptr,
                               std::vector<float *> &data_ptrs) {
  size_t const num_thread(24);
  std::vector<std::vector<std::vector<size_t>>> sub_src_idx_vecs(num_thread);
  std::vector<std::vector<std::vector<size_t>>> sub_dst_idx_vecs(num_thread);
  std::vector<std::vector<size_t>> sub_new_key_src_idx_vecs(num_thread);
#pragma omp parallel num_threads(num_thread)
  {
    size_t const tid(omp_get_thread_num());
    auto sub_chunk_size{len / num_thread};
    auto const idx{sub_chunk_size * tid};
    if (tid == num_thread - 1) sub_chunk_size += (len % num_thread);

    sub_src_idx_vecs[tid].resize(2);
    sub_dst_idx_vecs[tid].resize(2);
    for (size_t i{0}; i < 2; i++) {
      sub_src_idx_vecs[tid][i].reserve(sub_chunk_size);
      sub_dst_idx_vecs[tid][i].reserve(sub_chunk_size);
    }
    sub_new_key_src_idx_vecs[tid].reserve(sub_chunk_size);
    for (size_t i{idx}; i < idx + sub_chunk_size; i++) {
      auto key{key_ptr[i]};
      auto idx_pair{cascade_find_(key)};
      if (idx_pair.first == -1) {
        sub_new_key_src_idx_vecs[tid].push_back(i);
        continue;
      }
      sub_src_idx_vecs[tid][idx_pair.first].push_back(i);
      sub_dst_idx_vecs[tid][idx_pair.first].push_back(idx_pair.second);
    }
  }

  std::vector<size_t> new_key_src_idx_vec;
  std::vector<std::vector<size_t>> src_idx_vecs(2);
  std::vector<std::vector<size_t>> dst_idx_vecs(2);
#pragma omp parallel for num_threads(2)
  for (size_t i = 0; i < 2; i++) {
    size_t total_elem{0};
    for (size_t tid{0}; tid < num_thread; tid++) {
      total_elem += sub_src_idx_vecs[tid][i].size();
    }
    if (total_elem == 0) continue;
    src_idx_vecs[i].resize(total_elem);
    dst_idx_vecs[i].resize(total_elem);
    size_t offset{0};
    for (size_t tid{0}; tid < num_thread; tid++) {
      auto num_elem{sub_src_idx_vecs[tid][i].size()};
      if (num_elem == 0) continue;

      auto src_ptr{sub_src_idx_vecs[tid][i].data()};
      auto dst_ptr{src_idx_vecs[i].data() + offset};
      memcpy(dst_ptr, src_ptr, num_elem * sizeof(size_t));

      auto src_id_ptr{sub_dst_idx_vecs[tid][i].data()};
      auto dst_id_ptr{dst_idx_vecs[i].data() + offset};
      memcpy(dst_id_ptr, src_id_ptr, num_elem * sizeof(size_t));
      offset += num_elem;
    }
  }
  size_t total_elem{0};
  for (size_t tid{0}; tid < num_thread; tid++) {
    total_elem += sub_new_key_src_idx_vecs[tid].size();
  }
  new_key_src_idx_vec.resize(total_elem);
  size_t offset{0};
  for (size_t tid{0}; tid < num_thread; tid++) {
    auto num_elem{sub_new_key_src_idx_vecs[tid].size()};
    if (num_elem == 0) continue;

    auto src_id_ptr{sub_new_key_src_idx_vecs[tid].data()};
    auto dst_id_ptr{new_key_src_idx_vec.data() + offset};
    memcpy(dst_id_ptr, src_id_ptr, num_elem * sizeof(size_t));
    offset += num_elem;
  }

  omp_set_nested(2);
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp parallel for num_threads(24)
      for (size_t cnt = 0; cnt < src_idx_vecs[0].size(); cnt++) {
        auto src_idx{src_idx_vecs[0][cnt]};
        auto blk_idx{dst_idx_vecs[0][cnt] / block_capacity_};
        auto dst_idx{dst_idx_vecs[0][cnt] % block_capacity_};
        if (use_slot_id_) slot_ids_[blk_idx][dst_idx] = slot_id_ptr[src_idx];
        for (size_t i{0}; i < vec_per_line_; i++) {
          float *src_ptr{data_ptrs[i] + src_idx * emb_vec_size_};
          float *dst_ptr{cache_datas_[blk_idx][i].data() + dst_idx * emb_vec_size_};
          memcpy(dst_ptr, src_ptr, emb_vec_size_ * sizeof(float));
        }
      }
    }
#pragma omp section
    {
      sparse_model_file_ptr_->dump_update(dst_idx_vecs[1], src_idx_vecs[1], slot_id_ptr, data_ptrs);
    }
  }
  sparse_model_file_ptr_->dump_insert(key_ptr, new_key_src_idx_vec, slot_id_ptr, data_ptrs);
  {
    const double hit_rate{100.0 * src_idx_vecs[0].size() / len};
    HCTR_LOG_S(INFO, WORLD) << "HMEM-Cache PS: Hit rate [dump]: " << std::setprecision(4)
                            << hit_rate << " %" << std::endl;
  }
}

template <typename TypeKey>
void HMemCache<TypeKey>::sync_to_ssd() {
  if (!is_full_ && head_id_ == -1) return;
  auto num_blk{is_full_ ? num_block_ : (head_id_ + 1)};
  auto tail_id{is_full_ ? (head_id_ + 1) % num_block_ : 0};
  HCTR_LOG(INFO, ROOT, "Sync blocks from HMEM-Cache to SSD\n");
  tqdm bar;
  if (resource_manager_->is_master_process()) {
    bar.progress(0, num_blk);
  }
  for (auto cnt{0}; cnt < num_blk; cnt++) {
    auto blk_idx{(tail_id + cnt) % num_block_};
    sparse_model_file_ptr_->dump_update(key_idx_maps_[blk_idx], slot_ids_[blk_idx],
                                        cache_datas_[blk_idx]);
    if (resource_manager_->is_master_process()) {
      bar.progress(cnt + 1, num_blk);
    }
  }
  sparse_model_file_ptr_->update_global_model();
  bar.finish();
}

template class HMemCache<long long>;
template class HMemCache<unsigned>;

}  // namespace HugeCTR
