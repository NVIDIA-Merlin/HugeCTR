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
#include "HugeCTR/include/embeddings/embedding.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/model_oversubscriber/parameter_server_manager.hpp"

#include <memory>
#include <vector>
#include <typeinfo>

namespace HugeCTR {

class ModelOversubscriberImplBase {
public:
  virtual void store(std::vector<std::string> snapshot_file_list) = 0;
  virtual void update(std::vector<std::string>& keyset_file_list) = 0;
  virtual void update(std::string& keyset_file) = 0;
  virtual ~ModelOversubscriberImplBase() {}
};


template <typename TypeHashKey, typename TypeEmbeddingComp>
class ModelOversubscriberImpl : public ModelOversubscriberImplBase {
  std::vector<std::shared_ptr<IEmbedding>> embeddings_;
  ParameterServerManager<TypeHashKey, TypeEmbeddingComp> ps_manager_;

  size_t get_max_embedding_size_() {
    size_t max_embedding_size = 0;
    for (auto &one_embedding : embeddings_) {
      size_t embedding_size = one_embedding->get_max_vocabulary_size();
      max_embedding_size = (embedding_size > max_embedding_size) ?
        embedding_size : max_embedding_size;
    }
    return max_embedding_size;
  }

  /**
   * @brief      Load the embedding table according to keys stored in keyset_file_list from
   *             SSD to device memory.
   */
  void load_(std::vector<std::string>& keyset_file_list);

public:
  ModelOversubscriberImpl(
      std::vector<std::shared_ptr<IEmbedding>>& embeddings,
      const std::vector<SparseEmbeddingHashParams<TypeEmbeddingComp>>& embedding_params,
      const SolverParser& solver_config,
      const std::string& temp_embedding_dir);

  ModelOversubscriberImpl(const ModelOversubscriberImpl&) = delete;
  ModelOversubscriberImpl& operator=(const ModelOversubscriberImpl&) = delete;

  ~ModelOversubscriberImpl() {}

  /**
   * @brief      Store the embedding table or a snapshot file downloaded from device to SSD.
   *             If snapshot_file_list.size() is 0, update the embedding_file in SSD;
   *             Or, wirte out a snapshot.
   * @param      snapshot_file_list The file list where snapshot will be written, its size 
   *                                equals the number of embeddings.
   */
  void store(std::vector<std::string> snapshot_file_list = std::vector<std::string>()) override;

  /**
   * @brief      Updates the embedding_file using embeddings from device memory, then
   *             load embeddings to device memory according to the new keyset.
   * @param      keyset_file_list  The keyset file list storing keyset.
   */
  void update(std::vector<std::string>& keyset_file_list) override;

  /**
   * @brief      Updates the embedding_file using embeddings from device memory, then
   *             load embeddings to device memory according to the new keyset.
   * @param      keyset_file  A single keyset file storing keysets for all embeddings.
   */
  void update(std::string& keyset_file) override;
};

}  // namespace HugeCTR
