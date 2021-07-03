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

#include "tensor2.hpp"
#include "parser.hpp"
#include "HugeCTR/include/model_oversubscriber/parameter_server.hpp"

namespace HugeCTR {

template <typename TypeKey>
class ParameterServerManager {
  std::vector<std::shared_ptr<ParameterServer<TypeKey>>> ps_;
  BufferBag buf_bag_;

public:
  ParameterServerManager(bool use_host_ps,
      const std::vector<std::string>& sparse_embedding_files,
      const std::vector<Embedding_t>& embedding_types,
      const std::vector<SparseEmbeddingHashParams>& embedding_params,
      size_t buffer_size, std::shared_ptr<ResourceManager> resource_manager);

  ParameterServerManager(const ParameterServerManager&) = delete;
  ParameterServerManager& operator=(const ParameterServerManager&) = delete;

  auto get_parameter_server(int i) { return ps_[i]; }

  size_t get_size() { return ps_.size(); }

  BufferBag& get_buffer_bag() { return buf_bag_; }

  void update_sparse_model_file() {
    for (auto& ps : ps_) { ps->flush_emb_tbl_to_ssd(); }
  }
};

}  // namespace HugeCTR
