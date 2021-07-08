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

#include <embedding.hpp>
#include "HugeCTR/include/embeddings/embedding.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/model_oversubscriber/parameter_server_manager.hpp"

#include <algorithm>
#include <iterator>

namespace HugeCTR {

class ModelOversubscriberImplBase {
public:
  virtual void dump() = 0;
  virtual void update(std::vector<std::string>& keyset_file_list) = 0;
  virtual void update(std::string& keyset_file) = 0;
  virtual void update_sparse_model_file() = 0;
  virtual ~ModelOversubscriberImplBase() = default;
};


template <typename TypeKey>
class ModelOversubscriberImpl : public ModelOversubscriberImplBase {
  std::vector<std::shared_ptr<IEmbedding>> embeddings_;
  ParameterServerManager<TypeKey> ps_manager_;

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
   * @brief Load the embedding table according to keys stored in
   *        keyset_file_list from sparse_model_entity_ to device memory.
   */
  void load_(std::vector<std::string>& keyset_file_list);

public:
  ModelOversubscriberImpl(bool use_host_ps,
      std::vector<std::shared_ptr<IEmbedding>>& embeddings,
      const std::vector<SparseEmbeddingHashParams>& embedding_params,
      const std::vector<std::string>& sparse_embedding_files,
      std::shared_ptr<ResourceManager> resource_manager);

  ModelOversubscriberImpl(const ModelOversubscriberImpl&) = delete;
  ModelOversubscriberImpl& operator=(const ModelOversubscriberImpl&) = delete;

  ~ModelOversubscriberImpl() = default;

  /**
     * @brief Dump the downloaded embeddings from GPUs to sparse_model_entity_.
     */
    void dump() override;

  /**
   * @brief Updates the sparse_model_entity_ using embeddings from devices,
   *        then load embeddings to device memory according to the new keyset.
   * @param keyset_file_list The file list storing keyset files.
   */
  void update(std::vector<std::string>& keyset_file_list) override;

  /**
   * @brief Updates the sparse_model_entity_ using embeddings from devices,
   *        then load embeddings to device memory according to the new keyset.
   * @param keyset_file A single file storing keysets for all embeddings.
   */
  void update(std::string& keyset_file) override;

  void update_sparse_model_file() override {
    ps_manager_.update_sparse_model_file();
  }
};

}  // namespace HugeCTR
