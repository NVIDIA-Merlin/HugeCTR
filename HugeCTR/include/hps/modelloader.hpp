/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <hps/database_backend.hpp>
#include <io/filesystem.hpp>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace HugeCTR {

// This is a draft for a unified embedding format and needs to keep consistency with 3g embedding
// or merlinkv
template <typename TypeHashKey, typename TypeHashValue>
struct UnifiedEmbeddingTable {
  std::vector<TypeHashKey> keys;
  std::vector<TypeHashValue> vectors;
  std::vector<TypeHashKey> meta;
  std::vector<TypeHashKey> uvm_keys;
  std::vector<TypeHashValue> uvm_vectors;
  size_t key_count = 0;
  size_t vec_elem_count = 0;
  size_t total_key_count = 0;
  size_t threshold = -1;
  size_t cache_capacity = 0;
  size_t uvm_key_count = 0;
  void* get_cache_keys();
  void* get_caceh_vecs();
  size_t get_cache_key_count();
  void* get_uvm_keys();
  void* get_uvm_vecs();
  size_t get_uvm_key_count();
};

/**
 * Base interface for model loader. It is only used to encapsulate the logic of reading model files
 * in different formats, and does not keep any data members Implementations that inherit from this
 * should override all public members.
 *
 */
class IModelLoader {
 public:
  ~IModelLoader() = default;
  /**
   * @brief Returns all embedding keys and vectors for a specific number of iterations for cache and
   * uvm
   *
   * @param iteration
   * @param emb_size
   * @param cache_capacity
   */
  virtual void get_cache_uvm(size_t iteration, size_t emb_size, size_t cache_capacity) = 0;
  /**
   * Load the contents of the model file with difference format into a into a
   * UnifiedEmbeddingTable(data member) of an inherited class.
   *
   * @param table_name The destination table into which to insert the data.
   * @param path File system path under which the embedding file should be parsed.
   * @param key_num_per_iteration The number of key-value pairs that need to be parsed per iteration
   * @param threshold Threshold for filtering key-value pairs that need to be cached
   * iteration.
   */
  virtual void load(const std::string& table_name, const std::string& path,
                    size_t key_num_per_iteration = 0, size_t threshold = -1) = 0;
  /**
   * Load the contents of the model file with difference format into a into a
   * UnifiedEmbeddingTable(data member) of an inherited class.
   *
   * @param table_name The destination table into which to insert the data.
   * @param path_list File system path lists under which the multiple embedding folders should be
   * parsed.
   */
  virtual void load_fused_emb(const std::string& table_name,
                              const std::vector<std::string>& path_list) = 0;

  /**
   * free the UnifiedEmbeddingTable(data member) of an inherited class.
   *
   */
  virtual void delete_table() = 0;
  /**
   * Return the pointer of the embedding key from UnifiedEmbeddingTable.
   *
   */
  virtual void* getkeys() = 0;
  /**
   * Return the pointer of the embedding vectors from UnifiedEmbeddingTable.
   *
   */
  virtual void* getvectors() = 0;
  /**
   * Return the pointer of the meta info from UnifiedEmbeddingTable, such as scaler for
   * dequantization, timesample for each key
   *
   */
  virtual void* getmetas() = 0;
  /**
   * Return the number of embedding keys of current table
   *
   */
  virtual size_t getkeycount() = 0;
  /**
   * Returns the number of iterations to traverse all tables
   *
   */
  virtual size_t get_num_iterations() = 0;

  /**
   * @brief Returns all embedding keys for a specific number of iterations
   *
   * @param iteration
   */
  virtual std::pair<void*, size_t> getkeys(size_t iteration) = 0;
  /**
   * Returns all embedding vectors for a specific number of iterations
   *@param iteration
   *@param embedding_vector_size
   */
  virtual std::pair<void*, size_t> getvectors(size_t iteration, size_t emb_size) = 0;

  virtual void* get_cache_keys() = 0;
  virtual void* get_caceh_vecs() = 0;
  virtual size_t get_cache_key_count() = 0;
  virtual void* get_uvm_keys() = 0;
  virtual void* get_uvm_vecs() = 0;
  virtual size_t get_uvm_key_count() = 0;

  IModelLoader() = default;
};

/**
 * Implementations of read/parse embedding from legacy foramt model file, which is general format
 * for hugectr model file.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey, typename TValue>
class RawModelLoader : public IModelLoader {
 private:
  UnifiedEmbeddingTable<TKey, TValue>* embedding_table_;
  size_t num_iterations;
  std::unique_ptr<HugeCTR::FileSystem> fs_;
  size_t key_iteration;
  std::string embedding_folder_path;
  size_t key_num_iteration = 0;
  virtual void load_emb(const std::string& table_name, const std::string& path);

 public:
  RawModelLoader();
  virtual void load(const std::string& table_name, const std::string& path,
                    size_t key_num_per_iteration, size_t threshold);

  virtual void load_fused_emb(const std::string& table_name,
                              const std::vector<std::string>& path_list);
  virtual void delete_table();
  virtual void* getkeys();
  virtual void* getvectors();
  virtual void* getmetas();
  virtual void get_cache_uvm(size_t iteration, size_t emb_size, size_t cache_capacity);
  virtual size_t getkeycount();
  virtual size_t get_num_iterations();
  virtual std::pair<void*, size_t> getkeys(size_t iteration);
  virtual std::pair<void*, size_t> getvectors(size_t iteration, size_t emb_size);
  virtual void* get_cache_keys();
  virtual void* get_caceh_vecs();
  virtual size_t get_cache_key_count();
  virtual void* get_uvm_keys();
  virtual void* get_uvm_vecs();
  virtual size_t get_uvm_key_count();
  ~RawModelLoader() { delete_table(); }
};

template <typename TKey, typename TValue>
class ModelLoader {
 public:
  static IModelLoader* CreateLoader(DatabaseTableDumpFormat_t type) {
    switch (type) {
      case DatabaseTableDumpFormat_t::Raw:
        return new RawModelLoader<TKey, TValue>();
        break;
      // TBD: The load_dump logic implemented in the data backend can be encapsulated as another
      // model reader for sst/bin files, So as to facilitate the reuse by components, such as
      // embedding cache
      case DatabaseTableDumpFormat_t::SST:
        return nullptr;
        break;
      default:
        return NULL;
        break;
    }
  }
};

}  // namespace HugeCTR