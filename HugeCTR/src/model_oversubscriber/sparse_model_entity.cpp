/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "model_oversubscriber/sparse_model_entity.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <experimental/filesystem>
#include <omp.h>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

template <typename TypeKey>
SparseModelEntity<TypeKey>::SparseModelEntity(bool use_host_ps, 
    const std::string &sparse_model_file, Embedding_t embedding_type,
    size_t emb_vec_size, std::shared_ptr<ResourceManager> resource_manager)
  : use_host_ps_(use_host_ps),
    is_distributed_(embedding_type == Embedding_t::DistributedSlotSparseEmbeddingHash),
    emb_vec_size_(emb_vec_size), resource_manager_(resource_manager),
    sparse_model_file_(SparseModelFile<TypeKey>(sparse_model_file, embedding_type,
                                                emb_vec_size, resource_manager)) {
  if (use_host_ps_) {
    sparse_model_file_.load_emb_tbl_to_mem(exist_key_idx_mapping_, host_emb_tabel_);
  }
}

template <typename TypeKey>
void SparseModelEntity<TypeKey>::load_vec_by_key(std::vector<TypeKey>& keys,
    BufferBag &buf_bag, size_t& hit_size) {
  try {
    if (keys.empty()) {
      MESSAGE_("No keyset specified for loading");
      return;
    }
    TypeKey *key_ptr = Tensor2<TypeKey>::stretch_from(buf_bag.keys).get_ptr();
    float *vec_ptr = buf_bag.embedding.get_ptr();
    size_t *slot_id_ptr = nullptr;
    if (!is_distributed_) {
      slot_id_ptr = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    }

    if (use_host_ps_) {
      // load vectors from host memory
      std::vector<size_t> idx_exist;
      idx_exist.reserve(keys.size());
      size_t cnt_hit_keys = 0;

      auto insert_key_op = [this, &idx_exist, &cnt_hit_keys, &key_ptr, &slot_id_ptr](auto iter) {
        key_ptr[cnt_hit_keys] = iter->first;
        if (!is_distributed_) {
          slot_id_ptr[cnt_hit_keys] = iter->second.first;
        }
        idx_exist.push_back(iter->second.second);
        cnt_hit_keys++;
      };

      for (auto key : keys) {
        auto iter = exist_key_idx_mapping_.find(key);
        if (iter != exist_key_idx_mapping_.end()) {
          insert_key_op(iter);
        } else {
          iter = new_key_idx_mapping_.find(key);
          if (iter != new_key_idx_mapping_.end()) {
            insert_key_op(iter);
          }
        }
      }
      hit_size = cnt_hit_keys;

      #pragma omp parallel num_threads(8)
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
    } else {
      // load vectors from ssd
      std::vector<TypeKey> exist_keys;
      exist_keys.reserve(keys.size());
      
      const auto ssd_key_idx_mapping = sparse_model_file_.get_key_index_map();
      auto is_key_exist_op = [&ssd_key_idx_mapping](TypeKey key) {
        auto iter = ssd_key_idx_mapping.find(key);
        if (iter == ssd_key_idx_mapping.end()) return false;
        return true;
      };
      copy_if(keys.begin(), keys.end(), std::back_inserter(exist_keys), is_key_exist_op);
      hit_size = exist_keys.size();

      std::vector<size_t> slots;
      std::vector<float> vecs;
      sparse_model_file_.load_exist_vec_by_key(exist_keys, slots, vecs);

      memcpy(key_ptr, exist_keys.data(), exist_keys.size() * sizeof(TypeKey));
      memcpy(vec_ptr, vecs.data(), vecs.size() * sizeof(float));
      if (!is_distributed_) {
        memcpy(slot_id_ptr, slots.data(), slots.size() * sizeof(size_t));
      }
    }

#ifdef KEY_HIT_RATIO
    int my_rank = resource_manager_->get_process_id();

    std::stringstream ss;
    ss << "[Rank " << my_rank << "] loads " << keys.size() << " keys, hit "
       << hit_size << " (" << std::fixed << std::setprecision(2)
       << hit_size * 100.0 / keys.size() << "%) in existing model";
    MESSAGE_(ss.str(), true);
#endif
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
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
    if (use_host_ps_) {
      const size_t num_exist_vecs = host_emb_tabel_.size() / emb_vec_size_;

      std::vector<size_t> idx_dst;
      idx_dst.reserve(dump_size);

      for (size_t cnt = 0; cnt < dump_size; cnt++) {
        auto iter = exist_key_idx_mapping_.find(key_ptr[cnt]);
        if (iter != exist_key_idx_mapping_.end()) {
          idx_dst.push_back(iter->second.second);
          continue;
        }

        iter = new_key_idx_mapping_.find(key_ptr[cnt]);
        if (iter == new_key_idx_mapping_.end()) {
          size_t slot_id_temp = is_distributed_ ? 0 : slot_id_ptr[cnt];
          size_t vec_idx_temp = num_exist_vecs + cnt_new_keys++;
          new_key_idx_mapping_.emplace(key_ptr[cnt], std::make_pair(slot_id_temp, vec_idx_temp));
          idx_dst.push_back(vec_idx_temp);
        } else {
          idx_dst.push_back(iter->second.second);
        }
      }
      size_t extended_table_size = host_emb_tabel_.size() + cnt_new_keys * emb_vec_size_;
      host_emb_tabel_.resize(extended_table_size);

      #pragma omp parallel num_threads(8)
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
    } else {
      std::vector<TypeKey> exist_keys, new_keys;
      std::vector<size_t> exist_vec_idx, new_vec_idx;
      exist_keys.reserve(dump_size);
      exist_vec_idx.reserve(dump_size);
      new_keys.reserve(dump_size);
      new_vec_idx.reserve(dump_size);

      const auto ssd_key_idx_map = sparse_model_file_.get_key_index_map();
      for (size_t cnt = 0; cnt < dump_size; cnt++) {
        auto iter = ssd_key_idx_map.find(key_ptr[cnt]);
        if (iter != ssd_key_idx_map.end()) {
          exist_keys.push_back(key_ptr[cnt]);
          exist_vec_idx.push_back(cnt);
        } else {
          new_keys.push_back(key_ptr[cnt]);
          new_vec_idx.push_back(cnt);
        }
      }
      cnt_new_keys = new_keys.size();

#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
      int num_proc = resource_manager_->get_num_process();
      int my_rank = resource_manager_->get_process_id();
      for (int pid = 0; pid < num_proc; pid++) {
        if (my_rank == pid) {
#endif
          sparse_model_file_.dump_exist_vec_by_key(exist_keys, exist_vec_idx, vec_ptr);
          sparse_model_file_.append_new_vec_and_key(new_keys, slot_id_ptr, new_vec_idx, vec_ptr);
#ifdef ENABLE_MPI 
        }
        CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
      }
#endif
    }

#ifdef KEY_HIT_RATIO
    size_t num_hit = dump_size - cnt_new_keys;
    int my_rank = resource_manager_->get_process_id();

    std::stringstream ss;
    ss << "[Rank " << my_rank << "] dumps " << dump_size << " keys, hit "
       << num_hit << " (" << std::fixed << std::setprecision(2)
       << num_hit * 100.0 / dump_size << "%) in existing model";
    MESSAGE_(ss.str(), true);
#endif
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void SparseModelEntity<TypeKey>::flush_emb_tbl_to_ssd() {
  try {
    if (!use_host_ps_) return;
    MESSAGE_("Updating sparse model in SSD", false, false);

    std::vector<TypeKey> exist_keys, new_keys;
    std::vector<size_t> exist_vec_idx, new_vec_idx, new_slots;

    exist_keys.reserve(exist_key_idx_mapping_.size());
    exist_vec_idx.reserve(exist_key_idx_mapping_.size());
    new_keys.reserve(new_key_idx_mapping_.size());
    new_slots.reserve(new_key_idx_mapping_.size());
    new_vec_idx.reserve(new_key_idx_mapping_.size());

    for (const auto& exist_pair : exist_key_idx_mapping_) {
      exist_keys.push_back(exist_pair.first);
      exist_vec_idx.push_back(exist_pair.second.second);
    }
    for (const auto& new_pair : new_key_idx_mapping_) {
      new_keys.push_back(new_pair.first);
      new_slots.push_back(new_pair.second.first);
      new_vec_idx.push_back(new_pair.second.second);
    }

    exist_key_idx_mapping_.insert(new_key_idx_mapping_.begin(),
                                  new_key_idx_mapping_.end());
    new_key_idx_mapping_.clear();

#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
    int num_proc = resource_manager_->get_num_process();
    int my_rank = resource_manager_->get_process_id();
    for (int pid = 0; pid < num_proc; pid++) {
      if (my_rank == pid) {
#endif
        sparse_model_file_.dump_exist_vec_by_key(exist_keys, exist_vec_idx, host_emb_tabel_.data());
        sparse_model_file_.append_new_vec_and_key(new_keys, new_slots.data(), new_vec_idx, host_emb_tabel_.data());
#ifdef ENABLE_MPI 
      }
      CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
    }
#endif
    MESSAGE_(" [DONE]", false, true, false);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template class SparseModelEntity<long long>;
template class SparseModelEntity<unsigned>;

}  // namespace HugeCTR
