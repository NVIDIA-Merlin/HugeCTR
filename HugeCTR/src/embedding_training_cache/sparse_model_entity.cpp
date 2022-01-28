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

#include "embedding_training_cache/sparse_model_entity.hpp"

#include <omp.h>

#include <algorithm>
#include <execution>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>

#include "utils.hpp"

namespace {

using namespace HugeCTR;

template <typename TypeDst, typename TypeSrc>
void type_convert(std::vector<TypeDst> &dst_vec, const std::vector<TypeSrc> &src_vec) {
  dst_vec.resize(src_vec.size());
  std::transform(std::execution::par, src_vec.begin(), src_vec.end(), dst_vec.begin(),
                 [](TypeSrc key) { return static_cast<TypeDst>(key); });
}

template <typename TypeKey>
void parallel_table_lookup(
    const std::vector<TypeKey> &keys,
    const std::unordered_map<TypeKey, std::pair<size_t, size_t>> &exist_key_idx_mapping,
    const std::unordered_map<TypeKey, std::pair<size_t, size_t>> &new_key_idx_mapping,
    BufferBag &buf_bag, std::vector<TypeKey> &exist_keys, std::vector<size_t> &exist_idx,
    bool is_distributed, bool save_to_buf_bag) {
  TypeKey *key_ptr{nullptr};
  size_t *slot_id_ptr{nullptr};

  if (save_to_buf_bag) {
    key_ptr = Tensor2<TypeKey>::stretch_from(buf_bag.keys).get_ptr();
    if (!is_distributed) {
      slot_id_ptr = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    }
  }

  exist_idx.reserve(keys.size());

  const size_t chunk_num = std::thread::hardware_concurrency();
  std::vector<std::vector<TypeKey>> chunk_keys(chunk_num);
  std::vector<std::vector<size_t>> chunk_slot_id(chunk_num);
  std::vector<std::vector<size_t>> chunk_idx_exist(chunk_num);

#pragma omp parallel num_threads(chunk_num)
  {
    const size_t tid = omp_get_thread_num();
    const size_t thread_num = omp_get_num_threads();
    size_t sub_chunk_size = keys.size() / thread_num;
    size_t res_chunk_size = keys.size() % thread_num;
    const size_t idx = tid * sub_chunk_size;

    if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;
    chunk_keys[tid].reserve(sub_chunk_size);
    chunk_idx_exist[tid].reserve(sub_chunk_size);
    if (!is_distributed) chunk_slot_id[tid].reserve(sub_chunk_size);

    auto insert_key_op = [&](auto iter) {
      chunk_keys[tid].push_back(iter->first);
      chunk_idx_exist[tid].push_back(iter->second.second);
      if (!is_distributed) chunk_slot_id[tid].push_back(iter->second.first);
    };

    for (size_t i = 0; i < sub_chunk_size; i++) {
      auto key = keys[idx + i];
      auto iter = exist_key_idx_mapping.find(key);
      if (iter != exist_key_idx_mapping.end()) {
        insert_key_op(iter);
      } else {
        iter = new_key_idx_mapping.find(key);
        if (iter != new_key_idx_mapping.end()) {
          insert_key_op(iter);
        }
      }
    }
  }
  size_t cnt_hit_keys = 0;
  for (const auto &chunk_key : chunk_keys) {
    cnt_hit_keys += chunk_key.size();
  }

  exist_idx.resize(cnt_hit_keys);
  if (!save_to_buf_bag) {
    exist_keys.resize(cnt_hit_keys);
  }

  TypeKey *dst_key_ptr = save_to_buf_bag ? key_ptr : exist_keys.data();
  size_t *dst_slot_id_ptr = slot_id_ptr;
  size_t *dst_idx_exist_ptr = exist_idx.data();

  for (size_t i = 0; i < chunk_keys.size(); i++) {
    const auto num_elem = chunk_keys[i].size();

    memcpy(dst_key_ptr, chunk_keys[i].data(), num_elem * sizeof(TypeKey));
    dst_key_ptr += num_elem;

    memcpy(dst_idx_exist_ptr, chunk_idx_exist[i].data(), num_elem * sizeof(size_t));
    dst_idx_exist_ptr += num_elem;

    if (!is_distributed && save_to_buf_bag) {
      memcpy(dst_slot_id_ptr, chunk_slot_id[i].data(), num_elem * sizeof(size_t));
      dst_slot_id_ptr += num_elem;
    }
  }
}

}  // namespace

namespace HugeCTR {

template <typename TypeKey>
SparseModelEntity<TypeKey>::SparseModelEntity(const std::string &sparse_model_file,
                                              Embedding_t embedding_type, size_t emb_vec_size,
                                              std::shared_ptr<ResourceManager> resource_manager)
    : is_distributed_(embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash),
      emb_vec_size_(emb_vec_size),
      resource_manager_(resource_manager),
      sparse_model_file_(SparseModelFile<TypeKey>(sparse_model_file, embedding_type, emb_vec_size,
                                                  resource_manager)) {
  sparse_model_file_.load_emb_tbl_to_mem(exist_key_idx_mapping_, host_emb_tabel_);
}

template <typename TypeKey>
void SparseModelEntity<TypeKey>::load_vec_by_key(std::vector<TypeKey> &keys, BufferBag &buf_bag,
                                                 size_t &hit_size) {
  try {
    if (keys.empty()) {
      MESSAGE_("No keyset specified for loading");
      return;
    }
    float *vec_ptr = buf_bag.embedding.get_ptr();

    // load vectors from host memory
    std::vector<TypeKey> key_exist;
    std::vector<size_t> idx_exist;
    parallel_table_lookup(keys, exist_key_idx_mapping_, new_key_idx_mapping_, buf_bag, key_exist,
                          idx_exist, is_distributed_, true);
    hit_size = idx_exist.size();

#pragma omp parallel num_threads(std::thread::hardware_concurrency())
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = idx_exist.size() / thread_num;
      size_t res_chunk_size = idx_exist.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        size_t src_idx = idx_exist[idx + i] * emb_vec_size_;
        size_t dst_idx = (idx + i) * emb_vec_size_;
        memcpy(&vec_ptr[dst_idx], &host_emb_tabel_[src_idx], emb_vec_size_ * sizeof(float));
      }
    }

#ifdef KEY_HIT_RATIO
    std::stringstream ss;
    ss << "HMEM-PS: Load " << keys.size() << " keys, hit " << hit_size << " (" << std::fixed
       << std::setprecision(4) << hit_size * 100.0 / keys.size() << "%) in existing model";
    MESSAGE_(ss.str(), true);
#endif
  } catch (const internal_runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
std::pair<std::vector<long long>, std::vector<float>> SparseModelEntity<TypeKey>::load_vec_by_key(
    const std::vector<long long> &keys) {
  try {
    std::vector<TypeKey> key_to_search;
    type_convert(key_to_search, keys);

    BufferBag buf_bag_tmp;
    std::vector<TypeKey> key_exist;
    std::vector<size_t> idx_exist;
    parallel_table_lookup(key_to_search, exist_key_idx_mapping_, new_key_idx_mapping_, buf_bag_tmp,
                          key_exist, idx_exist, is_distributed_, false);

    auto hit_size = idx_exist.size();

    std::vector<long long> key_ll_exist;
    type_convert(key_ll_exist, key_exist);

    std::vector<float> emb_vectors(hit_size * emb_vec_size_);
    float *vec_ptr = emb_vectors.data();

#pragma omp parallel num_threads(std::thread::hardware_concurrency())
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = idx_exist.size() / thread_num;
      size_t res_chunk_size = idx_exist.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        size_t src_idx = idx_exist[idx + i] * emb_vec_size_;
        size_t dst_idx = (idx + i) * emb_vec_size_;
        memcpy(&vec_ptr[dst_idx], &host_emb_tabel_[src_idx], emb_vec_size_ * sizeof(float));
      }
    }

    return std::pair(std::move(key_ll_exist), std::move(emb_vectors));
  } catch (const internal_runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelEntity<TypeKey>::dump_vec_by_key(BufferBag &buf_bag, const size_t dump_size) {
  try {
    if (dump_size == 0) return;

    TypeKey *key_ptr = Tensor2<TypeKey>::stretch_from(buf_bag.keys).get_ptr();
    float *vec_ptr = buf_bag.embedding.get_ptr();
    size_t *slot_id_ptr = nullptr;
    if (!is_distributed_) {
      slot_id_ptr = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    }

    size_t cnt_new_keys = 0;
    const size_t num_exist_vecs = host_emb_tabel_.size() / emb_vec_size_;

    std::vector<size_t> idx_dst;
    idx_dst.resize(dump_size);

    const size_t chunk_num = std::thread::hardware_concurrency();
    std::vector<std::vector<int64_t>> chunk_idx_dst(chunk_num);
    std::vector<HashTableType> chunk_new_key_idx_mapping(chunk_num);
    std::vector<size_t> chunk_cnt_new_keys(chunk_num, 0);

#pragma omp parallel num_threads(chunk_num)
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = dump_size / thread_num;
      size_t res_chunk_size = dump_size % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;
      chunk_idx_dst[tid].reserve(sub_chunk_size);
      chunk_new_key_idx_mapping[tid].reserve(sub_chunk_size);

      for (size_t i = 0; i < sub_chunk_size; i++) {
        const auto key = key_ptr[idx + i];
        auto iter = exist_key_idx_mapping_.find(key);
        if (iter != exist_key_idx_mapping_.end()) {
          chunk_idx_dst[tid].push_back(iter->second.second);
          continue;
        }

        iter = new_key_idx_mapping_.find(key);
        if (iter == new_key_idx_mapping_.end()) {
          size_t slot_id_temp = is_distributed_ ? 0 : slot_id_ptr[idx + i];
          size_t vec_idx_temp = num_exist_vecs + chunk_cnt_new_keys[tid]++;
          chunk_new_key_idx_mapping[tid].emplace(key, std::make_pair(slot_id_temp, vec_idx_temp));
          chunk_idx_dst[tid].push_back(-1 * vec_idx_temp - 1);
        } else {
          chunk_idx_dst[tid].push_back(iter->second.second);
        }
      }
    }

    std::vector<size_t> new_key_offset(chunk_cnt_new_keys.size());
    std::exclusive_scan(chunk_cnt_new_keys.begin(), chunk_cnt_new_keys.end(),
                        new_key_offset.begin(), 0);

    std::vector<std::vector<size_t>> tmp_idx_dst(chunk_num);

    for (size_t tid = 0; tid < chunk_num; tid++) {
      tmp_idx_dst[tid].resize(chunk_idx_dst[tid].size());

#pragma omp parallel for num_threads(chunk_num)
      for (size_t i = 0; i < chunk_idx_dst[tid].size(); i++) {
        auto tmp_idx = chunk_idx_dst[tid][i];
        tmp_idx_dst[tid][i] = (tmp_idx < 0) ? -1 * (tmp_idx + 1) + new_key_offset[tid] : tmp_idx;
      }
    }

#pragma omp parallel for num_threads(chunk_num)
    for (size_t tid = 0; tid < chunk_num; tid++) {
      size_t offset = (tid == 0) ? 0 : dump_size / chunk_num * tid;
      memcpy(idx_dst.data() + offset, tmp_idx_dst[tid].data(),
             tmp_idx_dst[tid].size() * sizeof(size_t));

      for_each(std::execution::par, chunk_new_key_idx_mapping[tid].begin(),
               chunk_new_key_idx_mapping[tid].end(),
               [val = new_key_offset[tid]](auto &pair) { pair.second.second += val; });
    }

    cnt_new_keys = 0;
    for (size_t tid = 0; tid < chunk_num; tid++) {
      new_key_idx_mapping_.insert(chunk_new_key_idx_mapping[tid].begin(),
                                  chunk_new_key_idx_mapping[tid].end());

      cnt_new_keys += chunk_new_key_idx_mapping[tid].size();
    }

    size_t extended_table_size = host_emb_tabel_.size() + cnt_new_keys * emb_vec_size_;
    host_emb_tabel_.resize(extended_table_size);

#pragma omp parallel num_threads(chunk_num)
    {
      const size_t tid = omp_get_thread_num();
      const size_t thread_num = omp_get_num_threads();
      size_t sub_chunk_size = idx_dst.size() / thread_num;
      size_t res_chunk_size = idx_dst.size() % thread_num;
      const size_t idx = tid * sub_chunk_size;

      if (tid == thread_num - 1) sub_chunk_size += res_chunk_size;

      for (size_t i = 0; i < sub_chunk_size; i++) {
        size_t src_idx = (idx + i) * emb_vec_size_;
        size_t dst_idx = idx_dst[idx + i] * emb_vec_size_;
        memcpy(&host_emb_tabel_[dst_idx], &vec_ptr[src_idx], emb_vec_size_ * sizeof(float));
      }
    }

#ifdef KEY_HIT_RATIO
    size_t num_hit = dump_size - cnt_new_keys;

    std::stringstream ss;
    ss << "HMEM-PS: Dump " << dump_size << " keys, hit " << num_hit << " (" << std::fixed
       << std::setprecision(4) << num_hit * 100.0 / dump_size << "%) in existing model";
    MESSAGE_(ss.str(), true);
#endif
  } catch (const internal_runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelEntity<TypeKey>::flush_emb_tbl_to_ssd() {
  try {
    MESSAGE_("Updating sparse model in SSD", false, false);

    std::vector<TypeKey> exist_keys, new_keys;
    std::vector<size_t> exist_vec_idx, new_vec_idx, new_slots;

    exist_keys.reserve(exist_key_idx_mapping_.size());
    exist_vec_idx.reserve(exist_key_idx_mapping_.size());
    new_keys.reserve(new_key_idx_mapping_.size());
    new_slots.reserve(new_key_idx_mapping_.size());
    new_vec_idx.reserve(new_key_idx_mapping_.size());

    for (const auto &exist_pair : exist_key_idx_mapping_) {
      exist_keys.push_back(exist_pair.first);
      exist_vec_idx.push_back(exist_pair.second.second);
    }
    for (const auto &new_pair : new_key_idx_mapping_) {
      new_keys.push_back(new_pair.first);
      new_slots.push_back(new_pair.second.first);
      new_vec_idx.push_back(new_pair.second.second);
    }

    exist_key_idx_mapping_.insert(new_key_idx_mapping_.begin(), new_key_idx_mapping_.end());
    new_key_idx_mapping_.clear();

#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
    int num_proc = resource_manager_->get_num_process();
    int my_rank = resource_manager_->get_process_id();
    for (int pid = 0; pid < num_proc; pid++) {
      if (my_rank == pid) {
#endif
        sparse_model_file_.dump_exist_vec_by_key(exist_keys, exist_vec_idx, host_emb_tabel_.data());
        sparse_model_file_.append_new_vec_and_key(new_keys, new_slots.data(), new_vec_idx,
                                                  host_emb_tabel_.data());
#ifdef ENABLE_MPI
      }
      CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
    }
#endif
    MESSAGE_(" [DONE]", false, true, false);
  } catch (const internal_runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template class SparseModelEntity<long long>;
template class SparseModelEntity<unsigned>;

}  // namespace HugeCTR
