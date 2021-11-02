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
#include "model_oversubscriber/hmem_cache/hmem_cache.hpp"
#include "model_oversubscriber/sparse_model_entity.hpp"

namespace HugeCTR {

template <typename TypeKey>
class ParameterServer {
  TrainPSType_t ps_type_;
  bool use_slot_id_;
  std::unique_ptr<HMemCache<TypeKey>> hmem_cache_;
  std::unique_ptr<SparseModelEntity<TypeKey>> sparse_model_entity_;

  std::vector<TypeKey> keyset_;

 public:
  /**
   * @brief Constructs of ParameterServer. Using the sparse_model_file to
   *        initialize the sparse_model_entity_.
   * @param use_host_ps Whether to use the host memory-based parameter server.
   * @param sparse_model_file Folder name of the sparse embedding model.
   * @param embedding_type The type of embedding table object.
   * @param emb_vec_size Embedding vector size.
   * @param resource_manager The object of ResourceManager.
   */
  ParameterServer(TrainPSType_t ps_type, const std::string &sparse_model_file,
                  Embedding_t embedding_type, Optimizer_t opt_type, size_t emb_vec_size,
                  std::shared_ptr<ResourceManager> resource_manager, std::string local_path = "./",
                  HMemCacheConfig hmem_cache_config = HMemCacheConfig());

  ParameterServer(const ParameterServer &) = delete;
  ParameterServer &operator=(const ParameterServer &) = delete;

  ~ParameterServer() = default;

  /**
   * @brief Load the user-provided keyset from SSD, will be stored in keyset_.
   * @param keyset_file The file storing keyset to be loaded.
   */
  void load_keyset_from_file(std::string keyset_file);

  /**
   * @brief Pull embedding vectors from the sparse embedding model according to
   *        keyset_. It only loads embedding vectors that their corresponding
   *        keys exist in the trained sparse model.
   * @param buf_bag The buffer bag for keys, slot_id, and hash_table_val.
   * @param hit_size The number of keys to be loaded to be loaded to buffer bag.
   */
  void pull(BufferBag &buf_bag, size_t &hit_size);

  std::pair<std::vector<long long>, std::vector<float>> pull(
      const std::vector<long long> &keys_to_load);

  /**
   * @brief Push the embedding table downloaded from devices to the trained
   *        sparse model.
   * @param buf_bag The buffer bag for keys, slot_id, and hash_table_val.
   * @param dump_size The num of keys (features) in buffer bag to be dumped.
   */
  void push(BufferBag &buf_bag, size_t dump_size);

  /**
   * @brief Sync up the embedding table stored in SSD with the latest embedding
   *        table in the host memory.
   *        Note: The API will do nothing when use_host_ps = false.
   */
  void flush_emb_tbl_to_ssd();
};

}  // namespace HugeCTR
