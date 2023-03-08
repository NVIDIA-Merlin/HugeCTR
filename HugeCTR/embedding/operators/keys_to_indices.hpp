/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <HugeCTR/embedding/common.hpp>
#include <HugeCTR/embedding_storage/common.hpp>
#include <resource_manager.hpp>
namespace embedding {
class KeysToIndicesConverter {
 private:
  std::shared_ptr<core::CoreResourceManager> core_;

  std::vector<int> h_num_shards_;
  std::vector<int> h_local_table_ids_;

  core23::Tensor num_keys_per_table_offset_;
  core23::Tensor local_table_ids_;
  core23::Tensor num_shards_;

 public:
  KeysToIndicesConverter(std::shared_ptr<CoreResourceManager> core,
                         const std::vector<EmbeddingTableParam> &table_params,
                         const EmbeddingCollectionParam &ebc_param, size_t grouped_id);
  void convert(core23::Tensor &keys, size_t num_keys,
               const core23::Tensor &num_keys_per_lookup_offset, size_t num_lookups,
               const core23::Tensor &table_id_list);
};
}  // namespace embedding