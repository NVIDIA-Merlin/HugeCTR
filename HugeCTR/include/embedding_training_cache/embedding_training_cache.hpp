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
#include "embeddings/embedding_data.hpp"
#include "embedding_training_cache/hmem_cache/hmem_cache.hpp"
#include "embedding_training_cache/embedding_training_cache_impl.hpp"
#include "embedding_training_cache/parameter_server_manager.hpp"

namespace HugeCTR {

class EmbeddingTrainingCache {
 private:
  std::unique_ptr<EmbeddingTrainingCacheImplBase> impl_base_;

 public:
  EmbeddingTrainingCache(std::vector<TrainPSType_t> ps_types,
                      std::vector<std::shared_ptr<IEmbedding>> embeddings,
                      std::vector<std::string> sparse_embedding_files,
                      std::shared_ptr<ResourceManager> resource_manager, bool use_mixed_precision,
                      bool is_i64_key, std::vector<std::string> local_paths,
                      std::vector<HMemCacheConfig> hmem_cache_configs) {
    std::vector<SparseEmbeddingHashParams> embedding_params;
    if (is_i64_key) {
      for (auto& embedding : embeddings) {
        const auto& param = embedding->get_embedding_params();
        embedding_params.push_back(param);
      }
      impl_base_.reset(new EmbeddingTrainingCacheImpl<long long>(
          ps_types, embeddings, embedding_params, sparse_embedding_files, resource_manager,
          local_paths, hmem_cache_configs));
    } else {
      for (auto& embedding : embeddings) {
        const auto& param = embedding->get_embedding_params();
        embedding_params.push_back(param);
      }
      impl_base_.reset(new EmbeddingTrainingCacheImpl<unsigned>(
          ps_types, embeddings, embedding_params, sparse_embedding_files, resource_manager,
          local_paths, hmem_cache_configs));
    }
  }

  void dump() { impl_base_->dump(); }

  void update(std::vector<std::string>& keyset_file_list) { impl_base_->update(keyset_file_list); }

  void update(std::string& keyset_file) { impl_base_->update(keyset_file); }

  std::vector<std::pair<std::vector<long long>, std::vector<float>>> get_incremental_model(
      const std::vector<long long>& keys_to_load) {
    return impl_base_->get_incremental_model(keys_to_load);
  }

  void update_sparse_model_file() { impl_base_->update_sparse_model_file(); }
};

}  // namespace HugeCTR
