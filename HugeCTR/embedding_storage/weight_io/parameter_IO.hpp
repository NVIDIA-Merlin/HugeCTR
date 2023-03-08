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

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding_storage/weight_io/data_info.hpp>
#include <embedding_storage/weight_io/fs_interface.hpp>
#include <embeddings/embedding_collection.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace embedding {

class EmbeddingParameterIO {
 public:
  EmbeddingParameterIO() = default;

  EmbeddingParameterIO(const EmbeddingParameterIO&) = delete;

  EmbeddingParameterIO& operator=(const EmbeddingParameterIO&) = delete;

  EmbeddingParameterIO(std::shared_ptr<HugeCTR::ResourceManager> resource_manager);

  void add_embedding_collection(EmbeddingCollection* embedding_collection);

  void load_metadata(const std::string& parameters_folder_path, int ebc_id,
                     struct EmbeddingParameterInfo& epi);
  void load_embedding_weight(const struct EmbeddingParameterInfo& epi, int fs_table_id,
                             core23::Tensor& keys, core23::Tensor& embedding_weights,
                             embeddingFilter key_select,
                             std::shared_ptr<core::CoreResourceManager> core_resource,
                             const core23::DataType& target_key_type,
                             const core23::DataType& target_value_type);

  void load_opt_state(const struct EmbeddingParameterInfo& epi, int fs_table_id,
                      core23::Tensor& keys, core23::Tensor& optimizer_buffer,
                      embeddingFilter key_select,
                      std::shared_ptr<core::CoreResourceManager> core_resource,
                      const core23::DataType& target_key_type,
                      const core23::DataType& target_value_type);

  void get_parameter_info_from_model(const std::string& path,
                                     std::vector<struct EmbeddingParameterInfo>& epis);

  void dump_metadata(const std::string& parameters_folder_path,
                     const struct EmbeddingParameterInfo& epi,
                     const std::vector<int>& table_ids = std::vector<int>());

  void dump_embedding_weight(const std::string& parameters_folder_path,
                             struct EmbeddingParameterInfo& epi,
                             const std::vector<int>& table_ids = std::vector<int>());

  void dump_opt_state(const std::string& parameters_folder_path, struct EmbeddingParameterInfo& epi,
                      const std::vector<int>& table_ids = std::vector<int>());

  static std::shared_ptr<EmbeddingWeightIO> get_fs_object(
      const std::string& file_name, SparseFSType fs_type = SparseFSType::AUTO);

 private:
  void write_file_head(const std::string& path, EmbeddingFileType file_type, int table_id,
                       std::shared_ptr<EmbeddingWeightIO>& fs);

 private:
  std::vector<EmbeddingCollection*> embedding_collections_;
  HugeCTR::ResourceManager* resource_manager_ = nullptr;
  std::vector<std::shared_ptr<core::CoreResourceManager>> core_list_;
};

}  // namespace embedding
