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
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <hps/database_backend.hpp>
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
  std::vector<TypeHashValue> meta;
  size_t key_count = 0;
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
   * Load the contents of the model file with difference format into a into a
   * UnifiedEmbeddingTable(data member) of an inherited class.
   *
   * @param table_name The destination table into which to insert the data.
   * @param path File system path under which the embedding file should be parsed.
   */
  virtual void load(const std::string& table_name, const std::string& path) = 0;
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

 public:
  RawModelLoader();
  virtual void load(const std::string& table_name, const std::string& path);
  virtual void delete_table();
  virtual void* getkeys();
  virtual void* getvectors();
  virtual void* getmetas();
  virtual size_t getkeycount();
  ~RawModelLoader() { delete_table(); }
};

template <typename TKey, typename TValue>
class ModelLoader {
 public:
  static IModelLoader* CreateLoader(DBTableDumpFormat_t type) {
    switch (type) {
      case DBTableDumpFormat_t::Raw:
        return new RawModelLoader<TKey, TValue>();
        break;
      // TBD: The load_dump logic implemented in the data backend can be encapsulated as another
      // model reader for sst/bin files, So as to facilitate the reuse by components, such as
      // embedding cache
      case DBTableDumpFormat_t::SST:
        return nullptr;
        break;
      default:
        return NULL;
        break;
    }
  }
};

}  // namespace HugeCTR