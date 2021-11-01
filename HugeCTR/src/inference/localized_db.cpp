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

#include <inference/localized_db.hpp>
#include <iostream>

namespace HugeCTR {

template <typename TypeHashKey>
localdb<TypeHashKey>::localdb() {}

template <typename TypeHashKey>
localdb<TypeHashKey>::localdb(parameter_server_config ps_config) : ps_config_(ps_config) {}

template <typename TypeHashKey>
localdb<TypeHashKey>::~localdb() {
  if (cpu_embedding_table_.size() > 0) {
    cpu_embedding_table_.clear();
  }
}

template <typename TypeHashKey>
std::vector<std::vector<std::unordered_map<TypeHashKey, std::vector<float>>>>
localdb<TypeHashKey>::GetDB() {
  return cpu_embedding_table_;
}

template <typename TypeHashKey>
void localdb<TypeHashKey>::SetDB(
    std::vector<std::unordered_map<TypeHashKey, std::vector<float>>> model_emb_table) {
  cpu_embedding_table_.emplace_back(model_emb_table);
}

template <typename TypeHashKey>
void localdb<TypeHashKey>::load_data(std::vector<TypeHashKey> keys, std::vector<float> values,
                                     size_t embedding_size) {
  size_t num_key = keys.size();
  emb_table.reserve(num_key);
  for (size_t i = 0; i < num_key; i++) {
    emb_table.emplace(keys[i], std::vector<float>(values.begin() + i * embedding_size,
                                                  values.begin() + (i + 1) * embedding_size));
  }
  // Insert temp embedding table into temp model embedding table
  model_emb_table.emplace_back(emb_table);
}

template <typename TypeHashKey>
void localdb<TypeHashKey>::look_up(const TypeHashKey* embeddingcolumns_ptr, size_t num_samples,
                                   float* h_embeddingoutputvector, const std::string& model_name,
                                   size_t embedding_table_id) {
  if (num_samples <= 0) {
    return;
  }
  size_t model_id;
  auto model_id_iter = ps_config_.model_name_id_map_.find(model_name);
  if (model_id_iter != ps_config_.model_name_id_map_.end()) {
    model_id = model_id_iter->second;
  } else {
    std::cout << "Error: parameter server unknown model name. Note that this error will also come "
                 "out with "
                 "using Triton LOAD/UNLOAD APIs which haven't been supported in HugeCTR backend."
              << std::endl;
  }
  // Search for the embedding ids in the corresponding embedding table
  for (size_t i = 0; i < num_samples; i++) {
    // Look-up the id in the table
    auto result = cpu_embedding_table_[model_id][embedding_table_id].find(embeddingcolumns_ptr[i]);
    // Check if the key is existed in embedding table
    if (result != cpu_embedding_table_[model_id][embedding_table_id].end()) {
      // Find the embedding id
      size_t emb_vec_offset = i * ps_config_.embedding_vec_size_[model_id][embedding_table_id];
      memcpy(h_embeddingoutputvector + emb_vec_offset, (result->second).data(),
             sizeof(float) * ps_config_.embedding_vec_size_[model_id][embedding_table_id]);
    } else {
      // Cannot find the embedding id
      size_t emb_vec_offset = i * ps_config_.embedding_vec_size_[model_id][embedding_table_id];
      std::vector<float> default_emb_vec(
          ps_config_.embedding_vec_size_[model_id][embedding_table_id],
          ps_config_.default_emb_vec_value_[model_id][embedding_table_id]);
      memcpy(h_embeddingoutputvector + emb_vec_offset, default_emb_vec.data(),
             sizeof(float) * ps_config_.embedding_vec_size_[model_id][embedding_table_id]);
    }
  }
}

template class localdb<unsigned int>;
template class localdb<long long>;
}  // namespace HugeCTR
