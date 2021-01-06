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

#include <inference/parameter_server.hpp>

namespace HugeCTR {

template <typename TypeHashKey>
parameter_server<TypeHashKey>::parameter_server(const std::string& framework_name, 
                                                const std::vector<std::string>& model_config_path, 
                                                const std::vector<std::string>& model_name){
  // Store the configuration
  framework_name_ = framework_name;
  if(model_config_path.size() != model_name.size()){
    CK_THROW_(Error_t::WrongInput, "Wrong input: The size of input args are not consistent.");
  }
  // Initialize <model_name, id> map
  for(unsigned int i = 0; i < model_name.size(); i++){
    ps_config_.model_name_id_map_.emplace(model_name[i], (size_t)i);
  }
  
  // Initialize for each model
  for(unsigned int i = 0; i < model_config_path.size(); i++){
    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path[i]));

    // Read inference config
    const nlohmann::json& j_inference = get_json(model_config, "inference");
    const nlohmann::json& j_emb_table_file = get_json(j_inference, "sparse_model_file");
    std::vector<std::string> emb_file_path;
    if (j_emb_table_file.is_array()){
      for(unsigned int j = 0; j < j_emb_table_file.size(); j++){
        emb_file_path.emplace_back(j_emb_table_file[j].get<std::string>());
      }
    }
    else{
      emb_file_path.emplace_back(j_emb_table_file.get<std::string>());
    }
    ps_config_.emb_file_name_.emplace_back(emb_file_path);

    // Read embedding layer config
    const nlohmann::json& j_layers = get_json(model_config, "layers");
    std::vector<bool> distributed_emb;
    std::vector<size_t> embedding_vec_size;
    std::vector<float> default_emb_vec_value;
    // Search for all embedding layers
    for(unsigned int j = 1; j < j_layers.size(); j++){
      const nlohmann::json& j_single_layer = j_layers[j];
      std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
      if(embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0){
        distributed_emb.emplace_back(true);
        const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      }
      else if(embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 || embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0){
        distributed_emb.emplace_back(false);
        const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
        default_emb_vec_value.emplace_back(get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f));
      }
      else{
        break;
      }
    }
    ps_config_.distributed_emb_.emplace_back(distributed_emb);
    ps_config_.embedding_vec_size_.emplace_back(embedding_vec_size);
    ps_config_.default_emb_vec_value_.emplace_back(default_emb_vec_value);

  }

  if(ps_config_.distributed_emb_.size() != model_config_path.size() ||
     ps_config_.embedding_vec_size_.size() != model_config_path.size() ||
     ps_config_.default_emb_vec_value_.size() != model_config_path.size()){
    CK_THROW_(Error_t::WrongInput, "Wrong input: The size of parameter server parameters are not correct.");
  }

  // Load embeddings for each embedding table from each model
  for(unsigned int i = 0; i < model_config_path.size(); i++){
    size_t num_emb_table = (ps_config_.emb_file_name_[i]).size();
    // Temp vector of embedding table for this model
    std::vector<std::unordered_map<TypeHashKey, std::vector<float>>> model_emb_table;
    for(unsigned int j = 0; j < num_emb_table; j++){
      // Create input file stream to read the embedding file
      std::ifstream emb_file(ps_config_.emb_file_name_[i][j]);
      // Check if file is opened successfully
      if (!emb_file.is_open()){
        CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
      }
      emb_file.seekg(0, emb_file.end);
      size_t file_size = emb_file.tellg();
      emb_file.seekg(0, emb_file.beg);

      // The temp embedding table
      std::unordered_map<TypeHashKey, std::vector<float>> emb_table;

      if(ps_config_.distributed_emb_[i][j]){
        size_t row_size = sizeof(TypeHashKey) + sizeof(float) * ps_config_.embedding_vec_size_[i][j];
        size_t row_num = file_size / row_size;
        if (file_size % row_size != 0) {
          CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
        }

        TypeHashKey read_key;
        std::vector<float> read_emb_vec(ps_config_.embedding_vec_size_[i][j]);
        for(size_t pair = 0; pair < row_num; pair++){
          // Read out the emb_id and emb_vec
          emb_file.read(reinterpret_cast<char *>(&read_key), sizeof(TypeHashKey));
          emb_file.read(reinterpret_cast<char *>(read_emb_vec.data()),
                        sizeof(float) * ps_config_.embedding_vec_size_[i][j]);

          // Insert into CPU embedding table
          emb_table.emplace(read_key, read_emb_vec);
        }
      }
      else{
        size_t row_size = sizeof(TypeHashKey) + sizeof(size_t) + sizeof(float) * ps_config_.embedding_vec_size_[i][j];
        size_t row_num = file_size / row_size;
        if (file_size % row_size != 0){
          CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
        }

        TypeHashKey read_key;
        size_t read_slod_id;
        std::vector<float> read_emb_vec(ps_config_.embedding_vec_size_[i][j]);
        for(size_t pair = 0; pair < row_num; pair++){
          // Read out the emb_id, slot_id and emb_vec
          emb_file.read(reinterpret_cast<char *>(&read_key), sizeof(TypeHashKey));
          emb_file.read(reinterpret_cast<char *>(&read_slod_id), sizeof(size_t));
          emb_file.read(reinterpret_cast<char *>(read_emb_vec.data()),
                        sizeof(float) * ps_config_.embedding_vec_size_[i][j]);

          // Insert into CPU embedding table
          emb_table.emplace(read_key, read_emb_vec);
        }
      }

      // Insert temp embedding table into temp model embedding table
      model_emb_table.emplace_back(emb_table);
      // Close embedding file
      emb_file.close();
    }
    // Insert temp model embedding table into parameter server
    cpu_embedding_table_.emplace_back(model_emb_table);
  }
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::~parameter_server(){}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::look_up(const TypeHashKey* h_embeddingcolumns, 
                                            size_t length, 
                                            float* h_embeddingoutputvector, 
                                            const std::string& model_name, 
                                            size_t embedding_table_id){
  // Translate from model name to model id
  size_t model_id;
  auto model_id_iter = ps_config_.model_name_id_map_.find(model_name);
  if(model_id_iter != ps_config_.model_name_id_map_.end()){
    model_id = model_id_iter -> second;
  }
  else{
    CK_THROW_(Error_t::WrongInput, "Error: parameter server unknown model name.");
  }

  // Search for the embedding ids in the corresponding embedding table
  for(size_t i = 0; i < length; i++){
    // Look-up the id in the table
    auto result = cpu_embedding_table_[model_id][embedding_table_id].find(h_embeddingcolumns[i]);
    // Check if the key is existed in embedding table
    if(result != cpu_embedding_table_[model_id][embedding_table_id].end()){
      // Find the embedding id
      size_t emb_vec_offset = i * ps_config_.embedding_vec_size_[model_id][embedding_table_id];
      memcpy(h_embeddingoutputvector + emb_vec_offset, (result -> second).data(), sizeof(float) * ps_config_.embedding_vec_size_[model_id][embedding_table_id]);
    }
    else{
      // Cannot find the embedding id
      size_t emb_vec_offset = i * ps_config_.embedding_vec_size_[model_id][embedding_table_id];
      std::vector<float> default_emb_vec(ps_config_.embedding_vec_size_[model_id][embedding_table_id], ps_config_.default_emb_vec_value_[model_id][embedding_table_id]);
      memcpy(h_embeddingoutputvector + emb_vec_offset, default_emb_vec.data(), sizeof(float) * ps_config_.embedding_vec_size_[model_id][embedding_table_id]);
    }
  }
}

template class parameter_server<unsigned int>;
template class parameter_server<long long>;
}  // namespace HugeCTR
