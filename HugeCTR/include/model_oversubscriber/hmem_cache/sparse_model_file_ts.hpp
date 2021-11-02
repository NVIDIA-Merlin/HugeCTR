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

#include <memory>
#include <resource_manager.hpp>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

template <typename TypeKey>
class SparseModelFileTS {
 public:
  using HashTableType = std::unordered_map<TypeKey, size_t>;

 private:
  struct EmbeddingTableFile;

  struct MmapHandler {
    std::shared_ptr<EmbeddingTableFile> emb_tbl_;
    std::vector<float *> mmaped_ptrs_;
    bool mapped_to_file_{false};

    std::string get_folder_name() { return emb_tbl_->folder_name.c_str(); }
    std::string get_key_file() { return emb_tbl_->key_file; }
    std::string get_slot_file() { return emb_tbl_->slot_file; }
    std::vector<std::string> get_data_files() { return emb_tbl_->data_files; }
  };

  const std::string global_model_path_;
  const Optimizer_t opt_type_;

  MmapHandler mmap_handler_;
  HashTableType key_idx_map_;
  std::vector<size_t> slot_ids;

  bool mmap_valid_{false};

  const bool use_slot_id_;
  const size_t emb_vec_size_;

  std::shared_ptr<ResourceManager> resource_manager_;

  void mmap_to_memory_();
  void flush_mmap_to_disk_();
  void unmap_from_memory_();
  void expand_(size_t expand_size);

  static size_t num_instance;

 private:
  void update_local_model_();

 public:
  // end_flag should be the same defination with SoftwareCacheBase
  static size_t const end_flag{std::numeric_limits<size_t>::max()};
  SparseModelFileTS(std::string sparse_model_file, std::string local_path, bool use_slot_id,
                    Optimizer_t opt_type, size_t emb_vec_size,
                    std::shared_ptr<ResourceManager> resource_manager);

  ~SparseModelFileTS();

  __forceinline__ size_t find(TypeKey key) {
    auto it{key_idx_map_.find(key)};
    return ((it != key_idx_map_.end()) ? it->second : end_flag);
  }

  /**
   * @brief Load the embedding vector and optimizer state(s) by key.
   * @param key The key that its corresponding vector and opt state to be loaded.
   * @param slot_id The slot id corresponding to key. If use_slot_id is false,
   *                this parameter will be neglected.
   * @param data_ptrs A vector of buffers to store the embedding vector and opt
   *                  states. data_ptrs[0] is for the embedding vector, and
   *                  data_ptrs[1~data_ptrs.size()-1] are for the opt state(s).
   */
  void load(std::vector<size_t> const &ssd_idx_vec, size_t *slot_id_ptr,
            std::vector<float *> &data_ptrs);

  /**
   * @brief Dump the embedding vector and optimizer state(s) by key.
   * @param key The key that its corresponding vector and opt state to be dumped.
   * @param slot_id The slot id corresponding to key. If use_slot_id is false,
   *                this parameter will be neglected.
   * @param data_ptrs A vector of buffers storing the embedding vector and opt
   *                  states. data_ptrs[0] is for the embedding vector, and
   *                  data_ptrs[1~data_ptrs.size()-1] are for the opt state(s).
   */
  void dump_update(HashTableType &dump_key_idx_map, std::vector<size_t> &slot_id_vec,
                   std::vector<std::vector<float>> &data_vecs);

  void dump_update(std::vector<size_t> const &ssd_idx_vec, std::vector<size_t> const &mem_idx_vec,
                   size_t const *slot_id_ptr, std::vector<float *> &data_ptrs);

  void dump_insert(TypeKey const *key_ptr, std::vector<size_t> const &mem_src_idx,
                   size_t const *slot_id_ptr, std::vector<float *> &data_ptrs);

  void update_global_model();
};

}  // namespace HugeCTR
