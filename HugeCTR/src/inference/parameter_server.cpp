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
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

parameter_server_base::~parameter_server_base() {}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::parameter_server(const std::string& framework_name, 
                                              const std::vector<std::string>& model_config_path,
                                              const std::vector<InferenceParams>& inference_params_array){
  // Store the configuration
  framework_name_ = framework_name;
  if(model_config_path.size() != inference_params_array.size()){
    CK_THROW_(Error_t::WrongInput, "Wrong input: The size of input args are not consistent.");
  }
  // Initialize <model_name, id> map
  for(unsigned int i = 0; i < inference_params_array.size(); i++){
    ps_config_.model_name_id_map_.emplace(inference_params_array[i].model_name, (size_t)i);
  }
  
  // Initialize for each model
  for(unsigned int i = 0; i < model_config_path.size(); i++){
    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path[i]));

    // Read inference config
    std::vector<std::string> emb_file_path;
    if (inference_params_array[i].sparse_model_files.size() > 1){
      for(unsigned int j = 0; j < inference_params_array[i].sparse_model_files.size(); j++){
        emb_file_path.emplace_back(inference_params_array[i].sparse_model_files[j]);
      }
    }
    else{
      emb_file_path.emplace_back(inference_params_array[i].sparse_model_files[0]);
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

  auto remove_prefix = [](const std::string& path) {
    size_t found = path.rfind("/");
    if (found != std::string::npos)
      return std::string(path, found + 1);
    else
      return path;
  };

  // Load embeddings for each embedding table from each model
  for(unsigned int i = 0; i < model_config_path.size(); i++){
    size_t num_emb_table = (ps_config_.emb_file_name_[i]).size();
    // Temp vector of embedding table for this model
    std::vector<std::unordered_map<TypeHashKey, std::vector<float>>> model_emb_table;
    for(unsigned int j = 0; j < num_emb_table; j++){
      // Create input file stream to read the embedding file
      const std::string emb_file_prefix = ps_config_.emb_file_name_[i][j] + "/" +
                                          remove_prefix(ps_config_.emb_file_name_[i][j]);
      const std::string key_file = emb_file_prefix + ".key";
      const std::string vec_file = emb_file_prefix + ".vec";
      std::ifstream key_stream(key_file);
      std::ifstream vec_stream(vec_file);
      // Check if file is opened successfully
      if (!key_stream.is_open() || !vec_stream.is_open()){
        CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
      }
      size_t key_file_size_in_byte = fs::file_size(key_file);
      size_t vec_file_size_in_byte = fs::file_size(vec_file);

      size_t key_size_in_byte = sizeof(TypeHashKey);
      size_t vec_size_in_byte = sizeof(float) * ps_config_.embedding_vec_size_[i][j];

      size_t num_key = key_file_size_in_byte / key_size_in_byte;
      size_t num_vec = vec_file_size_in_byte / vec_size_in_byte;
      if (num_key != num_vec) {
        CK_THROW_(Error_t::WrongInput, "Error: num_key != num_vec in embedding file");
      }
      size_t num_float_val_in_vec_file = vec_file_size_in_byte / sizeof(float);

      // The temp embedding table
      std::vector<TypeHashKey> key_vec(num_key, 0);
      std::vector<float> vec_vec(num_float_val_in_vec_file, 0.0f);
      key_stream.read(reinterpret_cast<char *>(key_vec.data()), key_file_size_in_byte);
      vec_stream.read(reinterpret_cast<char *>(vec_vec.data()), vec_file_size_in_byte);

      std::unordered_map<TypeHashKey, std::vector<float>> emb_table;
      emb_table.reserve(num_key);
      const size_t emb_vec_size = ps_config_.embedding_vec_size_[i][j];
      for (size_t i = 0; i < num_key; i++) {
        emb_table.emplace(key_vec[i],
                          std::vector<float>(vec_vec.begin() + i     * emb_vec_size,
                                             vec_vec.begin() + (i+1) * emb_vec_size));
      }
      // Insert temp embedding table into temp model embedding table
      model_emb_table.emplace_back(emb_table);
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
    CK_THROW_(Error_t::WrongInput, "Error: parameter server unknown model name. Note that this error will also come out with using Triton LOAD/UNLOAD APIs which haven't been supported in HugeCTR backend.");
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
