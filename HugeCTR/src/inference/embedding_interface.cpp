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

#include <inference/embedding_cache.hpp>
#include <inference/embedding_interface.hpp>

namespace HugeCTR {

embedding_interface::embedding_interface() {}

embedding_interface::~embedding_interface() {}

template <typename TypeHashKey>
embedding_interface* embedding_interface::Create_Embedding_Cache(
    const std::string& model_config_path, const InferenceParams& inference_params,
    HugectrUtility<TypeHashKey>* parameter_server) {
  embedding_interface* new_embedding_cache;
  new_embedding_cache =
      new embedding_cache<TypeHashKey>(model_config_path, inference_params, parameter_server);
  return new_embedding_cache;
}

template embedding_interface* embedding_interface::Create_Embedding_Cache<unsigned int>(
    const std::string&, const InferenceParams&, HugectrUtility<unsigned int>*);
template embedding_interface* embedding_interface::Create_Embedding_Cache<long long>(
    const std::string&, const InferenceParams&, HugectrUtility<long long>*);
}  // namespace HugeCTR
