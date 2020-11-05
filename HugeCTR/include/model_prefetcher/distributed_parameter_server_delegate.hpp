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
#include <model_prefetcher/parameter_server_delegate.hpp>

namespace HugeCTR {

template <typename KeyType>
class DistributedParameterServerDelegate : public ParameterServerDelegate<KeyType> {
 public:
  using HashTable = typename ParameterServerDelegate<KeyType>::HashTable;

  void load(std::ofstream& embeding_table,
            std::ifstream& snapshot,
            const size_t file_size_in_byte,
            const size_t embedding_vector_size,
            HashTable& hash_table) override;
  void store(std::ofstream& snapshot,
             std::ifstream& embedding_table,
             const size_t file_size_in_byte,
             const size_t embedding_vector_size,
             HashTable& hash_table) override;
             
};

}  // namespace HugeCTR
