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

#include "embedding.hpp"
#include "model_oversubscriber/sparse_model_file.hpp"

namespace HugeCTR {

template <typename TypeKey>
class SparseModelEntity {
  using HashTableType = std::unordered_map<TypeKey, std::pair<size_t, size_t>>;

  bool use_host_ps_;
  std::vector<float> host_emb_tabel_;
  HashTableType exist_key_idx_mapping_;
  HashTableType new_key_idx_mapping_;
  bool is_distributed_;
  size_t emb_vec_size_;
  std::shared_ptr<ResourceManager> resource_manager_;
  SparseModelFile<TypeKey> sparse_model_file_;

 public:
  SparseModelEntity(bool use_host_ps, const std::string &sparse_model_file,
                    Embedding_t embedding_type, size_t emb_vec_size,
                    std::shared_ptr<ResourceManager> resource_manager);

  /**
   * @brief Load embedding features (embedding vectors) through provided keys
   *        either from disk (if use_host_ps_==false) or the host memory (if
   *        use_host_ps_==true). Some of the key in keys may not have
   *        corresponding embedding features, and they will be neglected.
   *
   * @param keys Vector stroing the keyset, their corresponding embedding
                 vectors will be loaded.
   * @param buf_bag A buffer bag to store the loaded key (slot_id if localized
                    embedding is used) and embedding vectors.
   * @param hit_size Number of keys that have corresponding embedding features.
   */
  void load_vec_by_key(std::vector<TypeKey> &keys, BufferBag &buf_bag, size_t &hit_size);

  std::pair<std::vector<long long>, std::vector<float>> load_vec_by_key(
      const std::vector<long long> &keys);

  /**
   * @brief Dump embedding features (embedding vectors) through provided keys to
   *        disk (if use_host_ps_==false) or the host memory (if use_host_ps_==
   *        true). Some of the keys may not exist in the sparse model, and they
   *        will be inserted after calling this API.
   *
   * @param buf_bag A buffer bag storing the key (slot_id if localized embedding
   *                is used) and embedding vectors, and they will be dumped.
   * @param dump_size Number of keys (slot_ids, embedding features) that are
   *                  going to be dumped.
   */
  void dump_vec_by_key(BufferBag &buf_bag, const size_t dump_size);

  /**
   * @brief Write the sparse model stored in the host memory to the disk. This
   *        function will do nothing when the SSD-based PS is used.
   */
  void flush_emb_tbl_to_ssd();
};

}  // namespace HugeCTR
