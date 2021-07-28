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
#include "HugeCTR/include/embeddings/embedding_data.hpp"
#include "HugeCTR/include/model_oversubscriber/parameter_server_manager.hpp"
#include "HugeCTR/include/model_oversubscriber/model_oversubscriber_impl.hpp"

namespace HugeCTR {

class ModelOversubscriber {
 private:
  std::unique_ptr<ModelOversubscriberImplBase> impl_base_;

public:
  ModelOversubscriber(bool use_host_ps,
      std::vector<std::shared_ptr<IEmbedding>>& embeddings,
      const std::vector<std::string>& sparse_embedding_files,
      std::shared_ptr<ResourceManager> resource_manager,
      bool use_mixed_precision, bool is_i64_key) {
    std::vector<SparseEmbeddingHashParams> embedding_params;
    if (is_i64_key) {
      for (auto& embedding : embeddings) {
        const auto& param = embedding->get_embedding_params();
        embedding_params.push_back(param);
      }
      impl_base_.reset(new ModelOversubscriberImpl<long long>(use_host_ps,
          embeddings, embedding_params, sparse_embedding_files, resource_manager));
    } else {
      for (auto& embedding : embeddings) {
        const auto& param = embedding->get_embedding_params();
        embedding_params.push_back(param);
      }
      impl_base_.reset(new ModelOversubscriberImpl<unsigned>(use_host_ps,
          embeddings, embedding_params, sparse_embedding_files, resource_manager));
    }
  }

  void dump() { impl_base_->dump(); }

  void update(std::vector<std::string>& keyset_file_list) { impl_base_->update(keyset_file_list); }

  void update(std::string& keyset_file) {
    impl_base_->update(keyset_file);
  }

  std::vector<std::pair<std::vector<long long>, std::vector<float>>>
      get_incremental_model(const std::vector<long long> &keys_to_load) {
    return impl_base_->get_incremental_model(keys_to_load);
  }

  void update_sparse_model_file() { impl_base_->update_sparse_model_file(); }
};

}  // namespace HugeCTR
