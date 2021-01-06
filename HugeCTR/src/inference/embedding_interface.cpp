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

#include <inference/embedding_interface.hpp>
#include <inference/embedding_cache.hpp>

namespace HugeCTR{

embedding_interface::embedding_interface(){}

embedding_interface::~embedding_interface(){}

template <typename TypeHashKey>
embedding_interface* embedding_interface::Create_Embedding_Cache(HugectrUtility<TypeHashKey>* parameter_server,
                                                                 int cuda_dev_id,
                                                                 bool use_gpu_embedding_cache,
                                                                 float cache_size_percentage,
                                                                 const std::string& model_config_path,
                                                                 const std::string& model_name){
  embedding_interface* new_embedding_cache;
  new_embedding_cache = new embedding_cache<TypeHashKey>(parameter_server, 
                                                         cuda_dev_id,
                                                         use_gpu_embedding_cache,
                                                         cache_size_percentage,
                                                         model_config_path,
                                                         model_name);
  return new_embedding_cache;
}

template embedding_interface* embedding_interface::Create_Embedding_Cache<unsigned int>(HugectrUtility<unsigned int>*,
                                                                                        int,
                                                                                        bool,
                                                                                        float,
                                                                                        const std::string&,
                                                                                        const std::string&);
template embedding_interface* embedding_interface::Create_Embedding_Cache<long long>(HugectrUtility<long long>*,
                                                                                        int,
                                                                                        bool,
                                                                                        float,
                                                                                        const std::string&,
                                                                                        const std::string&);
}  // namespace HugeCTR
