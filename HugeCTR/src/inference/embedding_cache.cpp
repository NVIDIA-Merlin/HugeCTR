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

#include <inference/embedding_cache.hpp>

namespace HugeCTR {

template <typename TypeHashKey>
embedding_cache<TypeHashKey>::embedding_cache(HugectrUtility<TypeHashKey>* parameter_server,
                                              int cuda_dev_id,
                                              bool use_gpu_embedding_cache,
                                              float cache_size_percentage,
                                              const std::string& model_config_path,
                                              const std::string& model_name){
  // Store the configuration
  parameter_server_ = parameter_server;
  cache_config_.cuda_dev_id_ = cuda_dev_id;
  cache_config_.cache_size_percentage_ = cache_size_percentage;
  cache_config_.use_gpu_embedding_cache_ = use_gpu_embedding_cache;
  cache_config_.model_name_ = model_name;

  // Open model config file and input model json config
  nlohmann::json model_config(read_json_file(model_config_path));
  // Read model json config
  const nlohmann::json& j_inference_layer = get_json(model_config, "inference");
  const nlohmann::json& j_emb_table_file = get_json(j_inference_layer, "sparse_model_file");
  if (j_emb_table_file.is_array()){
    cache_config_.num_emb_table_ = j_emb_table_file.size();
  }
  else{
    cache_config_.num_emb_table_ = 1;
  }
  
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::parameter_server(const std::string& model_name, const nlohmann::json& model_config) {
  model_name_ = model_name;
  // Parse parameters from json object
  const nlohmann::json& j_inference_layer = get_json(model_config, "inference");
  model_config_.emb_file_name_ = get_value_from_json<std::string>(j_inference_layer, "sparse_model_file");
  size_t max_sample_in_batch = get_value_from_json<size_t>(j_inference_layer, "max_batchsize");

  const nlohmann::json& j_layers_array = get_json(model_config, "layers");
  autconst nlohmann::json& j_data_layer = j_layers_array[0];
  std::string data_layer_type = get_value_from_json<std::string>(j_data_layer, "type");
  if(data_layer_type.compare("Data") != 0){
    CK_THROW_(Error_t::WrongInput, "The first layer is not Data layer:" + data_layer_type);
  }
  const nlohmann::json& j_data_sparse_array = get_json(j_data_layer, "sparse");
  // For now, only 1 embedding layer is supported
  size_t max_feature_num_per_sample = get_value_from_json<size_t>(j_data_sparse_array[0], "max_feature_num_per_sample");
  model_config_.max_query_length_ = max_feature_num_per_sample * max_sample_in_batch;
  
  // Search for all layers for embedding layer
  // For now, only 1 embedding layer is supported 
  for(unsigned int i = 1; i < j_layers_array.size(); i++) {
    const nlohmann::json& j_layer = j_layers_array[i];
    std::string embedding_name = get_value_from_json<std::string>(j_layer, "type");
    if(embedding_name.compare("DistributedSlotSparseEmbeddingHash") == 0){
      model_config_.distributed_emb_ = true;
      const nlohmann::json& embedding_hparam = get_json(j_layer, "sparse_embedding_hparam");
      model_config_.embedding_vec_size_ = get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size");
      model_config_.default_emb_vec_value_ = get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f);
      break;
    }
    else if(embedding_name.compare("LocalizedSlotSparseEmbeddingHash") == 0 || embedding_name.compare("LocalizedSlotSparseEmbeddingOneHot") == 0){
      model_config_.distributed_emb_ = false;
      const nlohmann::json& embedding_hparam = get_json(j_layer, "sparse_embedding_hparam");
      model_config_.embedding_vec_size_ = get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size");
      model_config_.default_emb_vec_value_ = get_value_from_json_soft<float>(embedding_hparam, "default_emb_vec_value", 0.0f);
      break;
    }
    else{
      CK_THROW_(Error_t::WrongInput, "No embedding layer found. ");
    }
  }

  //Allocate temp internal buffers
  CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingcolumns_, sizeof(TypeHashKey) * model_config_.max_query_length_, cudaHostAllocPortable));
  CK_CUDA_THROW_(cudaHostAlloc((void**)&h_embeddingoutputvector_, sizeof(float) * model_config_.embedding_vec_size_ * model_config_.max_query_length_, cudaHostAllocPortable));

  // Create input file stream to read the trained embeddings file
  std::ifstream emb_file(model_config_.emb_file_name_);

  // Check if file is opened successfully
  if (!emb_file.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Error: embeddings file not open for reading");
  }
  emb_file.seekg(0, emb_file.end);
  size_t file_size = emb_file.tellg();
  emb_file.seekg(0, emb_file.beg);

  // File format is different for distributed and localized embeddings trainning method
  if(model_config_.distributed_emb_){
    size_t row_size = sizeof(TypeHashKey) + sizeof(float) * model_config_.embedding_vec_size_;
    size_t row_num = file_size / row_size;
    if (file_size % row_size != 0) {
      CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
    }

    TypeHashKey read_key;
    std::vector<float> read_emb_vec(model_config_.embedding_vec_size_);
    for (size_t i = 0; i < row_num; i++) {
      // Read out the emb_id and emb_vec
      emb_file.read(reinterpret_cast<char *>(&read_key), sizeof(TypeHashKey));
      emb_file.read(reinterpret_cast<char *>(read_emb_vec.data()),
                    sizeof(float) * model_config_.embedding_vec_size_);

      // Insert into CPU embedding table
      cpu_embedding_table_.emplace(read_key, read_emb_vec);
    }
  }
  else{
    size_t row_size = sizeof(TypeHashKey) + sizeof(size_t) + sizeof(float) * model_config_.embedding_vec_size_;
    size_t row_num = file_size / row_size;
    if (file_size % row_size != 0) {
      CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
    }

    TypeHashKey read_key;
    size_t read_slod_id;
    std::vector<float> read_emb_vec(model_config_.embedding_vec_size_);
    for (size_t i = 0; i < row_num; i++) {
      // Read out the emb_id, slot_id and emb_vec
      emb_file.read(reinterpret_cast<char *>(&read_key), sizeof(TypeHashKey));
      emb_file.read(reinterpret_cast<char *>(&slod_id), sizeof(size_t));
      emb_file.read(reinterpret_cast<char *>(read_emb_vec.data()),
                    sizeof(float) * model_config_.embedding_vec_size_);

      // Insert into CPU embedding table
      cpu_embedding_table_.emplace(read_key, read_emb_vec);
    }
  }
}

template <typename TypeHashKey>
parameter_server<TypeHashKey>::~parameter_server() {
  // Free temp internal buffer
  CK_CUDA_THROW_(cudaFreeHost(h_embeddingcolumns_));
  CK_CUDA_THROW_(cudaFreeHost(h_embeddingoutputvector_));
}

template <typename TypeHashKey>
void parameter_server<TypeHashKey>::look_up(const TypeHashKey* embeddingcolumns, size_t length, float* embeddingoutputvector, cudaStream_t stream) {
  // Copy the embedding ids to internal host buffer
  CK_CUDA_THROW_(cudaMemcpyAsync(h_embeddingcolumns_, embeddingcolumns, sizeof(TypeHashKey) * length, cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));

  // Search for the embedding ids in the CPU embedding table
  for (size_t i = 0; i < length; i++) {
    // Look-up the id in the table
    auto result = cpu_embedding_table_.find(h_embeddingcolumns_[i]);
    // Check if the key is existed in embedding table
    if(result != cpu_embedding_table_.end()){
      // Find the embedding id
      size_t emb_vec_offset = i * model_config_.embedding_vec_size_;
      memcpy(h_embeddingoutputvector_ + emb_vec_offset, (result -> second).data(), sizeof(float) * model_config_.embedding_vec_size_);
    }
    else{
      // Cannot find the embedding id
      size_t emb_vec_offset = i * model_config_.embedding_vec_size_;
      std::vector<float> defualt_emb_vec(model_config_.embedding_vec_size_, model_config_.default_emb_vec_value_);
      memcpy(h_embeddingoutputvector_ + emb_vec_offset, defualt_emb_vec.data(), sizeof(float) * model_config_.embedding_vec_size_);
    }
  }

  // Copy the result back to device output buffer
  CK_CUDA_THROW_(cudaMemcpyAsync(embeddingoutputvector, h_embeddingoutputvector_, sizeof(float) * model_config_.embedding_vec_size_ * length, cudaMemcpyHostToDevice, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
}

}  // namespace HugeCTR
