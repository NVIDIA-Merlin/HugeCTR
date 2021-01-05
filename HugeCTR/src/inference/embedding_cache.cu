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
// Temp interface, should be delete later
nlohmann::json read_json_file(const std::string& filename);

// Kernels to combine the value buffer
__global__ void merge_emb_vec(
    float* d_output_emb_vec,
    const float* d_missing_emb_vec,
    const uint64_t* d_missing_index,
    const size_t len,
    const size_t emb_vec_size
    )
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < (len * emb_vec_size) )
    {
      size_t src_emb_vec = idx / emb_vec_size;
      size_t dst_emb_vec = d_missing_index[src_emb_vec];
      size_t dst_float = idx % emb_vec_size;
      d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
    }
}

template <typename TypeHashKey>
embedding_cache<TypeHashKey>::embedding_cache(HugectrUtility<TypeHashKey>* parameter_server,
                                              int cuda_dev_id,
                                              bool use_gpu_embedding_cache,
                                              float cache_size_percentage,
                                              const std::string& model_config_path,
                                              const std::string& model_name){
  // Store the configuration
  parameter_server_ = parameter_server;
  cache_config_.use_gpu_embedding_cache_ = use_gpu_embedding_cache;
  cache_config_.model_name_ = model_name;
  if(cache_config_.use_gpu_embedding_cache_){
    cache_config_.cuda_dev_id_ = cuda_dev_id;
    cache_config_.cache_size_percentage_ = cache_size_percentage;
  }

  // Open model config file and input model json config
  nlohmann::json model_config(read_json_file(model_config_path));

  // Read inference config
  const nlohmann::json& j_inference = get_json(model_config, "inference");
  const size_t max_batchsize = get_value_from_json<size_t>(j_inference, "max_batchsize");
  const nlohmann::json& j_emb_table_file = get_json(j_inference, "sparse_model_file");
  std::vector<std::string> emb_file_path;
  if (j_emb_table_file.is_array()){
    cache_config_.num_emb_table_ = j_emb_table_file.size();
    for(unsigned int i = 0; i < j_emb_table_file.size(); i++){
      emb_file_path.emplace_back(j_emb_table_file[i].get<std::string>());
    }
  }
  else{
    cache_config_.num_emb_table_ = 1;
    emb_file_path.emplace_back(j_emb_table_file.get<std::string>());
  }

  const nlohmann::json& j_layers = get_json(model_config, "layers");
  // Read data layer config
  const nlohmann::json& j_data_layer = j_layers[0];
  std::string data_layer_type = get_value_from_json<std::string>(j_data_layer, "type");
  if(data_layer_type.compare("Data") != 0){
    CK_THROW_(Error_t::WrongInput, "Wrong json format: The first layer is not Data layer:" + data_layer_type);
  }
  const nlohmann::json& j_data_layer_sparse_layer = get_json(j_data_layer, "sparse");
  if(!j_data_layer_sparse_layer.is_array()){
    CK_THROW_(Error_t::WrongInput, "Wrong json format: The sparse layer in data layer is not an array.");
  }
  if(j_data_layer_sparse_layer.size() != cache_config_.num_emb_table_){
    CK_THROW_(Error_t::WrongInput, "Wrong json format: The number of embedding table is not consistent.");
  }
  std::vector<size_t> max_feature_num_per_sample;
  for(unsigned int i = 0; i < j_data_layer_sparse_layer.size(); i++){
    max_feature_num_per_sample.emplace_back(get_value_from_json<size_t>(j_data_layer_sparse_layer[i], "max_feature_num_per_sample"));
  }

  // Read embedding layer config
  std::vector<bool> distributed_emb;
  // Search for all embedding layers
  for(unsigned int i = 1; i < j_layers.size(); i++){
    const nlohmann::json& j_single_layer = j_layers[i];
    std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
    if(embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0){
      distributed_emb.emplace_back(true);
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      cache_config_.embedding_vec_size_.emplace_back(get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
    }
    else if(embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 || embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0){
      distributed_emb.emplace_back(false);
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      cache_config_.embedding_vec_size_.emplace_back(get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
    }
    else{
      break;
    }
  }

  if(distributed_emb.size() != cache_config_.num_emb_table_){
    CK_THROW_(Error_t::WrongInput, "Wrong json format: The number of embedding table is not consistent.");
  }

  // Calculate max_query_len_per_emb_table
  for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
    cache_config_.max_query_len_per_emb_table_.emplace_back(max_batchsize * max_feature_num_per_sample[i]);
  }

  // Query the size of all embedding tables and calculate the size of each embedding cache
  if(cache_config_.use_gpu_embedding_cache_){
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      std::ifstream emb_file(emb_file_path[i]);
      // Check if file is opened successfully
      if (!emb_file.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Error: embeddings file cannot open for reading");
      }
      emb_file.seekg(0, emb_file.end);
      size_t file_size = emb_file.tellg();
      emb_file.seekg(0, emb_file.beg);

      // File format is different for distributed and localized embeddings
      if(distributed_emb[i]){
        size_t row_size = sizeof(TypeHashKey) + sizeof(float) * cache_config_.embedding_vec_size_[i];
        size_t row_num = file_size / row_size;
        if (file_size % row_size != 0){
          CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
        }
        size_t num_feature_in_cache = (size_t)((double)(cache_config_.cache_size_percentage_) * (double)row_num);
        cache_config_.num_set_in_cache_.emplace_back(num_feature_in_cache / (SLAB_SIZE * SET_ASSOCIATIVITY));
      }
      else{
        size_t row_size = sizeof(TypeHashKey) + sizeof(size_t) + sizeof(float) * cache_config_.embedding_vec_size_[i];
        size_t row_num = file_size / row_size;
        if (file_size % row_size != 0){
          CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
        }
        size_t num_feature_in_cache = (size_t)((double)(cache_config_.cache_size_percentage_) * (double)row_num);
        cache_config_.num_set_in_cache_.emplace_back(num_feature_in_cache / (SLAB_SIZE * SET_ASSOCIATIVITY));
      }
      emb_file.close();
    }
  }

  // Construct gpu embedding cache, 1 per embedding table
  if(cache_config_.use_gpu_embedding_cache_){

    // Device Restorer
    CudaDeviceContext dev_restorer;

    // Set CUDA device before creating gpu embedding cache
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      gpu_emb_caches_.emplace_back(new cache_(cache_config_.num_set_in_cache_[i], cache_config_.embedding_vec_size_[i]));
    }

  }
  
}

template <typename TypeHashKey>
embedding_cache<TypeHashKey>::~embedding_cache(){
  // Destruct gpu embedding cache
  if(cache_config_.use_gpu_embedding_cache_){
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before destructing gpu embedding cache
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      delete gpu_emb_caches_[i];
    }
  }
}

template <typename TypeHashKey> 
void embedding_cache<TypeHashKey>::look_up(const void* h_embeddingcolumns,
                                           const std::vector<size_t>& h_embedding_offset,
                                           float* d_shuffled_embeddingoutputvector,
                                           embedding_cache_workspace& workspace_handler,
                                           const std::vector<cudaStream_t>& streams){
  // Shuffle the input embeddingcolumns
  size_t num_sample = (h_embedding_offset.size() - 1) / cache_config_.num_emb_table_;
  size_t acc_offset = 0;
  for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
    workspace_handler.h_shuffled_embedding_offset_[i] = acc_offset;
    for(unsigned int j = 0; j < num_sample; j++){
      TypeHashKey* dst_ptr = (TypeHashKey*)(workspace_handler.h_shuffled_embeddingcolumns_) + acc_offset;
      TypeHashKey* src_prt = (TypeHashKey*)(h_embeddingcolumns) + h_embedding_offset[j * cache_config_.num_emb_table_ + i];
      size_t cpy_len = h_embedding_offset[j * cache_config_.num_emb_table_ + i + 1] - h_embedding_offset[j * cache_config_.num_emb_table_ + i];
      size_t cpy_len_in_byte = cpy_len * sizeof(TypeHashKey);
      memcpy(dst_ptr, src_prt, cpy_len_in_byte);
      acc_offset += cpy_len;
    }
  }
  workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] = acc_offset;
  if(workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] != h_embedding_offset[num_sample * cache_config_.num_emb_table_]){
    CK_THROW_(Error_t::WrongInput, "Error: embeddingcolumns buffer size is not consist before and after shuffle.");
  }

  // If GPU embedding cache is enabled
  if(cache_config_.use_gpu_embedding_cache_){

    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing look up
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    // Copy the shuffled embeddingcolumns buffer to device
    CK_CUDA_THROW_(cudaMemcpyAsync(workspace_handler.d_shuffled_embeddingcolumns_, 
                                   workspace_handler.h_shuffled_embeddingcolumns_, 
                                   workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] * sizeof(TypeHashKey), 
                                   cudaMemcpyHostToDevice, 
                                   streams[0]));
    CK_CUDA_THROW_(cudaStreamSynchronize(streams[0]));

    // Query the embeddingcolumns from GPU embedding cache & copy the missing length back
    size_t acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      TypeHashKey* d_query_key_ptr = (TypeHashKey*)(workspace_handler.d_shuffled_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      float* d_vals_retrieved_ptr = d_shuffled_embeddingoutputvector + acc_emb_vec_offset;
      uint64_t* d_missing_index_ptr = workspace_handler.d_missing_index_ + workspace_handler.h_shuffled_embedding_offset_[i];
      TypeHashKey* d_missing_key_ptr = (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];

      gpu_emb_caches_[i] -> Query(d_query_key_ptr, 
                                  query_length, 
                                  d_vals_retrieved_ptr, 
                                  d_missing_index_ptr, 
                                  d_missing_key_ptr, 
                                  workspace_handler.d_missing_length_ + i, 
                                  streams[i]);
      
      CK_CUDA_THROW_(cudaMemcpyAsync(workspace_handler.h_missing_length_ + i, workspace_handler.d_missing_length_ + i, sizeof(size_t), cudaMemcpyDeviceToHost, streams[i]));
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
    }

    // Copy the missing embeddingcolumns to host
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      TypeHashKey* d_missing_key_ptr = (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      TypeHashKey* h_missing_key_ptr = (TypeHashKey*)(workspace_handler.h_missing_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      CK_CUDA_THROW_(cudaMemcpyAsync(h_missing_key_ptr, d_missing_key_ptr, workspace_handler.h_missing_length_[i] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Query the missing embeddingcolumns from Parameter Server
    acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      TypeHashKey* h_missing_key_ptr = (TypeHashKey*)(workspace_handler.h_missing_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      parameter_server_ -> look_up(h_missing_key_ptr, workspace_handler.h_missing_length_[i], h_vals_retrieved_ptr, cache_config_.model_name_, i);
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
    }

    //Copy missing emb_vec to device 
    acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
      float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
      size_t missing_len_in_float = workspace_handler.h_missing_length_[i] * cache_config_.embedding_vec_size_[i];
      size_t missing_len_in_byte = missing_len_in_float * sizeof(float); 
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      CK_CUDA_THROW_(cudaMemcpyAsync(d_vals_retrieved_ptr, h_vals_retrieved_ptr, missing_len_in_byte, cudaMemcpyHostToDevice, streams[i]));
    }

    //Merge missing emb_vec into output
    acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
      float* d_vals_merge_dst_ptr = d_shuffled_embeddingoutputvector + acc_emb_vec_offset;
      uint64_t* d_missing_index_ptr = workspace_handler.d_missing_index_ + workspace_handler.h_shuffled_embedding_offset_[i];
      size_t missing_len_in_float = workspace_handler.h_missing_length_[i] * cache_config_.embedding_vec_size_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      //Wait for memory copy to complete
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_, 0, streams[i]>>>(d_vals_merge_dst_ptr, 
                                                                                                    d_vals_retrieved_ptr, 
                                                                                                    d_missing_index_ptr, 
                                                                                                    workspace_handler.h_missing_length_[i],
                                                                                                    cache_config_.embedding_vec_size_[i]);
    }
  }
  else{
    //Query the shuffled embeddingcolumns from Parameter Server & copy to device output buffer
    size_t acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      TypeHashKey* h_query_key_ptr = (TypeHashKey*)(workspace_handler.h_shuffled_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length_in_float = query_length * cache_config_.embedding_vec_size_[i];
      size_t query_length_in_byte = query_length_in_float * sizeof(float);
      float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
      float* d_vals_retrieved_ptr = d_shuffled_embeddingoutputvector + acc_emb_vec_offset;
      acc_emb_vec_offset += query_length_in_float;
      parameter_server_ -> look_up(h_query_key_ptr, query_length, h_vals_retrieved_ptr, cache_config_.model_name_, i);
      CK_CUDA_THROW_(cudaMemcpyAsync(d_vals_retrieved_ptr, h_vals_retrieved_ptr, query_length_in_byte, cudaMemcpyHostToDevice, streams[i]));
    }
  }

}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::update(embedding_cache_workspace& workspace_handler, 
                                          const std::vector<cudaStream_t>& streams){
  // If GPU embedding cache is enabled
  if(cache_config_.use_gpu_embedding_cache_){
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing update
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    size_t acc_emb_vec_offset = 0;
    for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
      TypeHashKey* d_missing_key_ptr = (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) + workspace_handler.h_shuffled_embedding_offset_[i];
      float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] - workspace_handler.h_shuffled_embedding_offset_[i];
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      gpu_emb_caches_[i] -> Replace(d_missing_key_ptr, 
                                    workspace_handler.h_missing_length_[i], 
                                    d_vals_retrieved_ptr, 
                                    streams[i]);
    }
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::create_workspace(embedding_cache_workspace& workspace_handler){
  size_t max_query_len_per_batch = 0;
  size_t max_emb_vec_len_per_batch_in_float = 0;
  for(unsigned int i = 0; i < cache_config_.num_emb_table_; i++){
    max_query_len_per_batch += cache_config_.max_query_len_per_emb_table_[i];
    max_emb_vec_len_per_batch_in_float += (cache_config_.max_query_len_per_emb_table_[i] * cache_config_.embedding_vec_size_[i]);
  }
  // Allocate common buffer
  CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_shuffled_embeddingcolumns_, 
                               max_query_len_per_batch * sizeof(TypeHashKey), 
                               cudaHostAllocPortable));
  CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_shuffled_embedding_offset_, 
                               (cache_config_.num_emb_table_ + 1) * sizeof(size_t), 
                               cudaHostAllocPortable));
  CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_missing_emb_vec_, 
                               max_emb_vec_len_per_batch_in_float * sizeof(float), 
                               cudaHostAllocPortable));

  // If GPU embedding cache is enabled
  if(cache_config_.use_gpu_embedding_cache_){
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before creating workspace buffer
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_shuffled_embeddingcolumns_, 
                              max_query_len_per_batch * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_missing_embeddingcolumns_, 
                              max_query_len_per_batch * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_missing_embeddingcolumns_, 
                                 max_query_len_per_batch * sizeof(TypeHashKey), 
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_missing_length_, 
                              cache_config_.num_emb_table_ * sizeof(size_t)));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_missing_length_, 
                                 cache_config_.num_emb_table_ * sizeof(size_t), 
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_missing_index_, 
                              max_query_len_per_batch * sizeof(uint64_t)));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_missing_emb_vec_, 
                              max_emb_vec_len_per_batch_in_float * sizeof(float)));
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::destroy_workspace(embedding_cache_workspace& workspace_handler){
  // Free common buffer
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_shuffled_embeddingcolumns_));
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_shuffled_embedding_offset_));
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_emb_vec_));
  // If GPU embedding cache is enabled
  if(cache_config_.use_gpu_embedding_cache_){
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before free workspace buffer
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_shuffled_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_length_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_length_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_index_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_emb_vec_));
  }
}

template class embedding_cache<unsigned int>;
template class embedding_cache<long long>;
}  // namespace HugeCTR
