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
class SparseModelFile {
  struct EmbeddingTableFile;
  struct MmapHandler {
    std::shared_ptr<EmbeddingTableFile> emb_tbl_;
    float* mmaped_table_{nullptr};
    bool maped_to_memory_{false};
    const char* get_folder_name() { return emb_tbl_->folder_name.c_str(); }
    const char* get_key_file() { return emb_tbl_->key_file.c_str(); }
    const char* get_vec_file() { return emb_tbl_->vec_file.c_str(); }
    const char* get_slot_file() { return emb_tbl_->slot_file.c_str(); }
  };

  using HashTableType = std::unordered_map<TypeKey, std::pair<size_t, size_t>>;

  MmapHandler mmap_handler_;
  HashTableType key_idx_map_;
  bool is_distributed_;
  size_t emb_vec_size_;
  std::shared_ptr<ResourceManager> resource_manager_;

  void map_embedding_to_memory_();
  void sync_mmaped_embedding_with_disk_();
  void unmap_embedding_from_memory_();

 public:
  SparseModelFile(const std::string& sparse_model_file, Embedding_t embedding_type,
                  size_t emb_vec_size, std::shared_ptr<ResourceManager> resource_manager);

  HashTableType& get_key_index_map() { return key_idx_map_; }

  /**
   * @brief Load embedding features (embedding vectors) through provided keys from disk.
   *        The keyset stored in keys (and corresponding embedding vectors) must exist in
   *        the embedding file stored in disk. Or, a run-time error will be thrown out.
   *        This API can be called by multiple processors at the same time because reading
   *        a file by multiple processors through mmap simultaneous is safe.
   *
   * @param keys Vector stroing the keyset, their corresponding embedding vectors will be loaded.
   * @param slots Vector to store the loaded slot_id. It will be ignored when using
   * DistributedEmbedding.
   * @param vecs Vector to store the loaded embedding vectors.
   */
  void load_exist_vec_by_key(const std::vector<TypeKey>& keys, std::vector<size_t>& slots,
                             std::vector<float>& vecs);

  /**
   * @brief Dump embedding features (embedding vectors) through provided keys to disk.
   *        The keyset stored in keys (and corresponding embedding vectors) must exist in
   *        the embedding file stored in disk. Or, a run-time error will be thrown out.
   *        This API can only be called by a single processor each time because updating
   *        a file by multiple processors through mmap simultaneous will cause unexpected results.
   *
   * @param keys Vector storing the keyset, their corresponding embedding vectors will be dumped.
   * @param vec_indices The memory indices of vectors in vecs. These indices are corresponding to
   *                    embedding vectors mapping by keys.
   * @param vecs Array storing the embedding vectors to be dumped.
   */
  void dump_exist_vec_by_key(const std::vector<TypeKey>& keys,
                             const std::vector<size_t>& vec_indices, const float* vecs);

  /**
   * @brief Append <key, emb_vector> (distributed embedding) or <key, slot_id, emb_vector> (
   *        localized embedding) to disk.
   *        The keyset stored in keys (and corresponding slot_ids and embedding vectors) must
   *        don't exist in the embedding file stored in disk. It's user's responsibility to
   *        ensure this assumption.
   *        This API can only be called by a single processor each time because
   *        dump_exist_vec_by_key is called in this API.
   *
   * @param keys Vector storing the keyset, their corresponding embedding vectors (and slot_ids
   *             if localized embedding is used) will be dumped.
   * @param slots Array storing the slot_id to be dumped. It will be ignored when using
   *              DistributedEmbedding.
   * @param vec_indices The memory indices of vectors in vecs. These indices are corresponding to
   *                    embedding vectors mapping by keys.
   * @param vecs Array storing the embedding vectors to be dumped.
   */
  void append_new_vec_and_key(const std::vector<TypeKey>& keys, const size_t* slots,
                              const std::vector<size_t>& vec_indices, const float* vecs);

  /**
   * @brief Load the embedding table (<key, emb_vector> for distributed embedding and
   *        <key, slot_id, emb_vec> for localized embedding) from disk to the host memory.
   *        After calling this, user will have a host memory based embedding table.
   *        This API can be called by multiple processors at the same time because each rank
   *        only load the embedding table belonging to itself.
   *
   * @param mem_key_index_map The constructed key<-->slot_id<-->mem_idx mapping of the host
   *                          memory based embedding table.
   * @param vecs Vector to store the loaded embedding vectors corresponding to mem_key_index_map.
   */
  void load_emb_tbl_to_mem(HashTableType& mem_key_index_map, std::vector<float>& vecs);
};

}  // namespace HugeCTR
