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

#include <tensor2.hpp>
#include "HugeCTR/include/model_oversubscriber/parameter_server.hpp"

#include <vector>
#include <memory>
#include <parser.hpp>

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
class ParameterServerManager {
  std::vector<std::shared_ptr<ParameterServer<TypeHashKey, TypeEmbeddingComp>>> ps_;
  BufferBag buf_bag_;

public:
  ParameterServerManager(
      const std::vector<SparseEmbeddingHashParams<TypeEmbeddingComp>>& embedding_params,
      const Embedding_t embedding_type,
      const SolverParser& solver_config,
      const std::string& temp_embedding_dir,
      size_t buffer_size);

  ParameterServerManager(const ParameterServerManager&) = delete;
  ParameterServerManager& operator=(const ParameterServerManager&) = delete;

  ~ParameterServerManager() {}
  /**
   * @brief      Gets the ith parameter server by index.
   * @param      i     index of parameter server.
   * @return     shared pointer of the ith parameter server.
   */
  const std::shared_ptr<ParameterServer<TypeHashKey, TypeEmbeddingComp>> get_parameter_server(int i) {
    return ps_[i];
  }

  size_t get_size() { return ps_.size(); }

  BufferBag& get_buffer_bag() { return buf_bag_; }
};

}  // namespace HugeCTR
