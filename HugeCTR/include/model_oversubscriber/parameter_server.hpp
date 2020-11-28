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

#pragma once

#include <embedding.hpp>
#include <model_oversubscriber/parameter_server_delegate.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
class ParameterServer {
  SparseEmbeddingHashParams<TypeEmbeddingComp> embedding_params_;
  std::string embedding_table_path_;
  std::unique_ptr<ParameterServerDelegate<TypeHashKey>> parameter_server_delegate_;
  std::unordered_map<TypeHashKey, size_t> hash_table_;
  std::vector<TypeHashKey> keyset_;

  size_t file_size_in_byte_; /**< Size of embedding file in bytes */
  float* mmaped_table_;      /**< Memory mapped file pointer */
  int fd_;                   /**< File descriptor for mapped file */
  bool maped_to_memory_;
  
  void map_embedding_to_memory_();
  void unmap_embedding_from_memory_();

public:
  /**
   * @brief      Constructs of ParameterServer. Using the snapshot_src_file to
   *             initialize hash_table_ and a temporary embedding_file.
   * @param      embedding_params  The embedding parameters for initializetion.
   * @param      snapshot_src_file The source file used to initialize hash_table_
   *             and embedding_file.
   */
  ParameterServer(
      const SparseEmbeddingHashParams<TypeEmbeddingComp>& embedding_params,
      const std::string& snapshot_src_file,
      const std::string& temp_embedding_dir);

  ParameterServer(const ParameterServer&) = delete;
  ParameterServer& operator=(const ParameterServer&) = delete;

  ~ParameterServer();

  /**
   * @brief      Load the user-provided keyset from SSD, will be stored in keyset_.
   * @param      num_unique_key  The number unique keys stored in keyset_file.
   */
  void load_keyset_from_file(std::string keyset_file);
  
  /**
   * @brief      Load embedding vectors from SSD according to keyset_. It only loads embedding
   *             vectors that their corresponding keys exist in hash_table, return pointers of
   *             loaded embedding vectors and their corresponding keys.
   * @param      hash_table_val  embedding vectors loaded from SSD, the num of vecs is hit_size.
   * @param      keys      The keys corresponding to hash_table_val, and they exist in hash_table_.
   * @param      hit_size  The number of keys in keys.
   */
  void load_param_from_embedding_file(float* hash_table_val, TypeHashKey* keys, size_t* hit_size);

  /**
   * @brief      Dumps the embedding table to the embedding file.
   * @param      hash_table_val  The embedding table downloaded from device memory.
   * @param      keys            The keys corresponding to the downloaded embedding table.
   * @param      dump_size       The size of keys in buffer keys or vectors in embedding_table.
   */
  void dump_param_to_embedding_file(float* hash_table_val, TypeHashKey* keys, size_t dump_size);

  /**
   * @brief      Dump to snapshot_dst_file in a format of <key embedding_vector> for
   *             DistributedSlotSparseEmbedding, or <key slot embedding_vector> for
   *             LocalizedSlotSparseEmbedding.
   */
  void dump_to_snapshot(const std::string& snapshot_dst_file);

  /**
   * @brief      A function for debugging purpose, returning the keys from hash_table.
   */
  std::vector<TypeHashKey> get_keys_from_hash_table() const;
};

}  // namespace HugeCTR
