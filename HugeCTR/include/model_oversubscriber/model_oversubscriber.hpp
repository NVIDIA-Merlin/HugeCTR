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
#include "HugeCTR/include/model_oversubscriber/parameter_server_manager.hpp"
#include "HugeCTR/include/model_oversubscriber/model_oversubscriber_impl.hpp"

#include <memory>
#include <vector>

namespace HugeCTR {

class ModelOversubscriber {
private:
  std::unique_ptr<ModelOversubscriberImplBase> impl_base_;

public:
  ModelOversubscriber(
    std::vector<std::shared_ptr<IEmbedding>>& embeddings,
    std::vector<SparseEmbeddingHashParams>& embedding_params,
    const std::vector<std::string>& sparse_embedding_files, const Solver& solver) {
    if (solver.i64_input_key) {
      for (auto& one_embedding : embeddings) {
        embedding_params.push_back(
            dynamic_cast<Embedding<long long, float>*>(one_embedding.get())
                ->get_embedding_params());
      }
      impl_base_.reset(new ModelOversubscriberImpl<long long>(
          embeddings, embedding_params, sparse_embedding_files));
    } else {
      for (auto& one_embedding : embeddings) {
        embedding_params.push_back(
            dynamic_cast<Embedding<unsigned, float>*>(one_embedding.get())
                ->get_embedding_params());
      }
      impl_base_.reset(new ModelOversubscriberImpl<unsigned>(
          embeddings, embedding_params, sparse_embedding_files));
    }
  }

  void store() {
    impl_base_->store();
  }

  void update(std::vector<std::string>& keyset_file_list) {
    impl_base_->update(keyset_file_list);
  }

  void update(std::string& keyset_file) {
    impl_base_->update(keyset_file);
  }
};

}  // namespace HugeCTR
