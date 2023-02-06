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

#include <embedding_training_cache/hmem_cache/hmem_cache.hpp>
#include <embedding_training_cache/parameter_server.hpp>
#include <parser.hpp>
#include <tensor2.hpp>

namespace HugeCTR {

template <typename TypeKey>
class ParameterServerManager {
  std::vector<std::shared_ptr<ParameterServer<TypeKey>>> ps_;
  BufferBag buf_bag_;

 public:
  ParameterServerManager(std::vector<TrainPSType_t>& ps_types,
                         std::vector<std::string>& sparse_embedding_files,
                         std::vector<Embedding_t> embedding_types,
                         std::vector<SparseEmbeddingHashParams>& embedding_params,
                         size_t buffer_size, std::shared_ptr<ResourceManager> resource_manager,
                         std::vector<std::string>& local_paths,
                         std::vector<HMemCacheConfig>& hmem_cache_configs);

  ParameterServerManager(const ParameterServerManager&) = delete;
  ParameterServerManager& operator=(const ParameterServerManager&) = delete;

  auto get_parameter_server(int i) { return ps_[i]; }

  size_t get_size() { return ps_.size(); }

  BufferBag& get_buffer_bag() { return buf_bag_; }

  void update_sparse_model_file() {
    for (auto& ps : ps_) ps->flush_emb_tbl_to_ssd();
  }
};

}  // namespace HugeCTR
