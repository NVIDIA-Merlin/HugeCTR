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

#include <algorithm>
#include <experimental/filesystem>
#include <inference/embedding_cache.hpp>
#include <mutex>
#include <thread>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
                         const uint64_t* d_missing_index_ptr, const size_t missing_len,
                         const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream);

void fill_default_emb_vec_async(float* d_vals_merge_dst_ptr, const float default_emb_vec,
                                const uint64_t* d_missing_index_ptr, const size_t missing_len,
                                const size_t emb_vec_size, const size_t BLOCK_SIZE,
                                cudaStream_t stream);

void decompress_emb_vec_async(const float* d_unique_src_ptr, const uint64_t* d_unique_index_ptr,
                              float* d_decompress_dst_ptr, const size_t decompress_len,
                              const size_t emb_vec_size, const size_t BLOCK_SIZE,
                              cudaStream_t stream);

template <typename TypeHashKey>
static void parameter_server_insert_thread_func_(HugectrUtility<TypeHashKey>* parameter_server,
                                                 embedding_interface* embedding_cache,
                                                 MemoryBlock* memory_block,
                                                 embedding_cache_config& cache_config,
                                                 const std::vector<cudaStream_t>& streams,
                                                 std::mutex& mutex) {
  try {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<cudaEvent_t> insert_events;
    for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
      cudaEvent_t event;
      cudaEventCreate(&event);
      insert_events.push_back(event);
    }
    parameter_server->insert_embedding_cache(embedding_cache, cache_config,
                                             memory_block->worker_buffer, streams);
    for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
      cudaEventRecord(insert_events[i], streams[i]);
    }
    for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
      cudaEventSynchronize(insert_events[i]);
    }
    parameter_server->FreeBuffer(memory_block);
    for (unsigned int i = 0; i < cache_config.num_emb_table_; i++) {
      cudaEventDestroy(insert_events[i]);
    }
  } catch (const std::runtime_error& rt_err) {
    parameter_server->FreeBuffer(memory_block);
    std::cerr << rt_err.what() << std::endl;
  }
}

template <typename TypeHashKey>
embedding_cache<TypeHashKey>::embedding_cache(const std::string& model_config_path,
                                              const InferenceParams& inference_params,
                                              HugectrUtility<TypeHashKey>* parameter_server) {
  auto b2s = [](const char val) { return val ? "True" : "False"; };
  HCTR_LOG(INFO, ROOT, "Use GPU embedding cache: %s, cache size percentage: %f\n",
           b2s(inference_params.use_gpu_embedding_cache), inference_params.cache_size_percentage);
  HCTR_LOG(INFO, ROOT, "Configured cache hit rate threshold: %f\n",
           inference_params.hit_rate_threshold);
  // Store the configuration
  parameter_server_ = parameter_server;
  cache_config_.use_gpu_embedding_cache_ = inference_params.use_gpu_embedding_cache;
  cache_config_.default_value_for_each_table = inference_params.default_value_for_each_table;
  cache_config_.model_name_ = inference_params.model_name;
  if (cache_config_.use_gpu_embedding_cache_) {
    cache_config_.cuda_dev_id_ = inference_params.device_id;
    cache_config_.cache_refresh_percentage_per_iteration =
        inference_params.cache_refresh_percentage_per_iteration;
    cache_config_.cache_size_percentage_ = inference_params.cache_size_percentage;
  }

  // Open model config file and input model json config
  nlohmann::json model_config(read_json_file(model_config_path));

  // Read inference config
  const size_t max_batchsize = inference_params.max_batchsize;
  std::vector<std::string> emb_file_path;
  if (inference_params.sparse_model_files.size() > 1) {
    cache_config_.num_emb_table_ = inference_params.sparse_model_files.size();
    for (unsigned int i = 0; i < inference_params.sparse_model_files.size(); i++) {
      emb_file_path.emplace_back(inference_params.sparse_model_files[i]);
    }
  } else {
    cache_config_.num_emb_table_ = 1;
    emb_file_path.emplace_back(inference_params.sparse_model_files[0]);
  }

  const nlohmann::json& j_layers = get_json(model_config, "layers");
  // Read data layer config
  const nlohmann::json& j_data_layer = j_layers[0];
  std::string data_layer_type = get_value_from_json<std::string>(j_data_layer, "type");
  if (data_layer_type.compare("Data") != 0) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong json format: The first layer is not Data layer:" + data_layer_type);
  }
  const nlohmann::json& j_data_layer_sparse_layer = get_json(j_data_layer, "sparse");
  if (!j_data_layer_sparse_layer.is_array()) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong json format: The sparse layer in data layer is not an array.");
  }
  if (j_data_layer_sparse_layer.size() != cache_config_.num_emb_table_) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong json format: The number of embedding table is not consistent.");
  }
  std::vector<size_t> max_feature_num_per_sample;
  for (unsigned int i = 0; i < j_data_layer_sparse_layer.size(); i++) {
    max_feature_num_per_sample.emplace_back(static_cast<size_t>(
        get_max_feature_num_per_sample_from_nnz_per_slot(j_data_layer_sparse_layer[i])));
  }

  // Read embedding layer config
  std::vector<bool> distributed_emb;
  // Search for all embedding layers
  for (unsigned int i = 1; i < j_layers.size(); i++) {
    const nlohmann::json& j_single_layer = j_layers[i];
    std::string embedding_type = get_value_from_json<std::string>(j_single_layer, "type");
    if (embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
      distributed_emb.emplace_back(true);
      cache_config_.embedding_table_name_.emplace_back(
          get_value_from_json<std::string>(j_single_layer, "top"));
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      cache_config_.embedding_vec_size_.emplace_back(
          get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
    } else if (embedding_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
               embedding_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
      distributed_emb.emplace_back(false);
      cache_config_.embedding_table_name_.emplace_back(
          get_value_from_json<std::string>(j_single_layer, "top"));
      const nlohmann::json& embedding_hparam = get_json(j_single_layer, "sparse_embedding_hparam");
      cache_config_.embedding_vec_size_.emplace_back(
          get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
    } else {
      break;
    }
  }

  if (distributed_emb.size() != cache_config_.num_emb_table_) {
    CK_THROW_(Error_t::WrongInput,
              "Wrong json format: The number of embedding table is not consistent.");
  }

  // Calculate max_query_len_per_emb_table
  for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
    cache_config_.max_query_len_per_emb_table_.emplace_back(max_batchsize *
                                                            max_feature_num_per_sample[i]);
  }

  // Query the size of all embedding tables and calculate the size of each embedding cache
  if (cache_config_.use_gpu_embedding_cache_) {
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      std::string key_file(emb_file_path[i] + "/key");
      size_t row_num = fs::file_size(key_file) / sizeof(long long);
      if (fs::file_size(key_file) % sizeof(long long) != 0) {
        CK_THROW_(Error_t::WrongInput, "Error: embeddings file size is not correct");
      }

      size_t num_feature_in_cache =
          (size_t)((double)(cache_config_.cache_size_percentage_) * (double)row_num);
      cache_config_.num_set_in_cache_.emplace_back(num_feature_in_cache /
                                                   (SLAB_SIZE * SET_ASSOCIATIVITY));
    }
  }

  // Construct gpu embedding cache, 1 per embedding table
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;

    // Set CUDA device before creating gpu embedding cache
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      gpu_emb_caches_.emplace_back(
          new cache_(cache_config_.num_set_in_cache_[i], cache_config_.embedding_vec_size_[i]));
    }
  }

  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;

    // Set CUDA device before creating gpu embedding cache
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      insert_streams_.push_back(stream);
    }

    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      refresh_streams_.push_back(stream);
    }
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::finalize() {
  if (cache_config_.use_gpu_embedding_cache_) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before creating gpu embedding cache
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    // Join insert threads
    for (auto& ps_insert_thread : ps_insert_threads_) {
      if (ps_insert_thread.joinable()) {
        ps_insert_thread.join();
      }
    }
  }
}

template <typename TypeHashKey>
embedding_cache<TypeHashKey>::~embedding_cache() {
  // Destruct gpu embedding cache
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before destructing gpu embedding cache
    cudaSetDevice(cache_config_.cuda_dev_id_);
    for (auto& stream : insert_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& stream : refresh_streams_) {
      cudaStreamDestroy(stream);
    }
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      delete gpu_emb_caches_[i];
    }
  }
}

template <typename TypeHashKey>
bool embedding_cache<TypeHashKey>::look_up(const void* h_embeddingcolumns,
                                           const std::vector<size_t>& h_embedding_offset,
                                           float* d_shuffled_embeddingoutputvector,
                                           MemoryBlock* memory_block,
                                           const std::vector<cudaStream_t>& streams,
                                           float hit_rate_threshold) {
  embedding_cache_workspace workspace_handler = memory_block->worker_buffer;
  // Shuffle the input embeddingcolumns
  size_t num_sample = (h_embedding_offset.size() - 1) / cache_config_.num_emb_table_;
  size_t acc_offset = 0;
  for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
    workspace_handler.h_shuffled_embedding_offset_[i] = acc_offset;
    for (unsigned int j = 0; j < num_sample; j++) {
      TypeHashKey* dst_ptr =
          (TypeHashKey*)(workspace_handler.h_shuffled_embeddingcolumns_) + acc_offset;
      TypeHashKey* src_prt = (TypeHashKey*)(h_embeddingcolumns) +
                             h_embedding_offset[j * cache_config_.num_emb_table_ + i];
      size_t cpy_len = h_embedding_offset[j * cache_config_.num_emb_table_ + i + 1] -
                       h_embedding_offset[j * cache_config_.num_emb_table_ + i];
      size_t cpy_len_in_byte = cpy_len * sizeof(TypeHashKey);
      memcpy(dst_ptr, src_prt, cpy_len_in_byte);
      acc_offset += cpy_len;
    }
  }
  workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] = acc_offset;
  if (workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] !=
      h_embedding_offset[num_sample * cache_config_.num_emb_table_]) {
    CK_THROW_(Error_t::WrongInput,
              "Error: embeddingcolumns buffer size is not consist before and after shuffle.");
  }

  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing look up
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    // Copy the shuffled embeddingcolumns buffer to device
    CK_CUDA_THROW_(cudaMemcpyAsync(
        workspace_handler.d_shuffled_embeddingcolumns_,
        workspace_handler.h_shuffled_embeddingcolumns_,
        workspace_handler.h_shuffled_embedding_offset_[cache_config_.num_emb_table_] *
            sizeof(TypeHashKey),
        cudaMemcpyHostToDevice, streams[0]));
    CK_CUDA_THROW_(cudaStreamSynchronize(streams[0]));

    // De-duplicate the emb_id to each embedding cache(table), copy the length to host
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      TypeHashKey* d_input_key_ptr =
          (TypeHashKey*)(workspace_handler.d_shuffled_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      size_t unique_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                             workspace_handler.h_shuffled_embedding_offset_[i];
      uint64_t* d_output_index_ptr = workspace_handler.d_unique_output_index_ +
                                     workspace_handler.h_shuffled_embedding_offset_[i];
      TypeHashKey* d_unique_output_key_ptr =
          (TypeHashKey*)(workspace_handler.d_unique_output_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      size_t* d_output_counter_ptr = workspace_handler.d_unique_length_ + i;
      ((unique_op_*)(workspace_handler.unique_op_obj_[i]))
          ->unique(d_input_key_ptr, unique_length, d_output_index_ptr, d_unique_output_key_ptr,
                   d_output_counter_ptr, streams[i]);
      CK_CUDA_THROW_(cudaMemcpyAsync(workspace_handler.h_unique_length_ + i, d_output_counter_ptr,
                                     sizeof(size_t), cudaMemcpyDeviceToHost, streams[i]));
    }

    // Query the embeddingcolumns from GPU embedding cache & copy the missing length back
    size_t acc_emb_vec_offset = 0;
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      TypeHashKey* d_query_key_ptr =
          (TypeHashKey*)(workspace_handler.d_unique_output_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      float* d_vals_retrieved_ptr = workspace_handler.d_hit_emb_vec_ + acc_emb_vec_offset;
      uint64_t* d_missing_index_ptr =
          workspace_handler.d_missing_index_ + workspace_handler.h_shuffled_embedding_offset_[i];
      TypeHashKey* d_missing_key_ptr =
          (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      size_t query_length = workspace_handler.h_unique_length_[i];
      size_t task_per_warp_tile = (query_length < 1000000) ? 1 : 32;
      gpu_emb_caches_[i]->Query(d_query_key_ptr, query_length, d_vals_retrieved_ptr,
                                d_missing_index_ptr, d_missing_key_ptr,
                                workspace_handler.d_missing_length_ + i, streams[i],
                                task_per_warp_tile);

      CK_CUDA_THROW_(cudaMemcpyAsync(workspace_handler.h_missing_length_ + i,
                                     workspace_handler.d_missing_length_ + i, sizeof(size_t),
                                     cudaMemcpyDeviceToHost, streams[i]));
      acc_emb_vec_offset += (workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                             workspace_handler.h_shuffled_embedding_offset_[i]) *
                            cache_config_.embedding_vec_size_[i];
    }

    bool async_insert_flag = true;
    // Calculate the hit rate of this look_up
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
      if (workspace_handler.h_unique_length_[i] == 0) {
        workspace_handler.h_hit_rate_[i] = 1.0;
      } else {
        workspace_handler.h_hit_rate_[i] = 1.0 - ((double)(workspace_handler.h_missing_length_[i]) /
                                                  (double)(workspace_handler.h_unique_length_[i]));
      }
      async_insert_flag =
          async_insert_flag && (workspace_handler.h_hit_rate_[i] >= hit_rate_threshold);
    }

    // Handle the missing keys, mode 1: synchronous
    if (!async_insert_flag) {
      parameter_server_->insert_embedding_cache(this, cache_config_, memory_block->worker_buffer,
                                                streams);
      // Merge missing emb_vec into hit emb_vec buffer
      acc_emb_vec_offset = 0;
      for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
        float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
        float* d_vals_merge_dst_ptr = workspace_handler.d_hit_emb_vec_ + acc_emb_vec_offset;
        uint64_t* d_missing_index_ptr =
            workspace_handler.d_missing_index_ + workspace_handler.h_shuffled_embedding_offset_[i];
        size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                              workspace_handler.h_shuffled_embedding_offset_[i];
        acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
        // Wait for memory copy to complete
        CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
        merge_emb_vec_async(d_vals_merge_dst_ptr, d_vals_retrieved_ptr, d_missing_index_ptr,
                            workspace_handler.h_missing_length_[i],
                            cache_config_.embedding_vec_size_[i], BLOCK_SIZE_, streams[i]);
      }
    }
    // mode 2: Asynchronous
    else {
      acc_emb_vec_offset = 0;
      for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
        // Fill the default emb vec into output buffer
        float* d_vals_merge_dst_ptr = workspace_handler.d_hit_emb_vec_ + acc_emb_vec_offset;
        uint64_t* d_missing_index_ptr =
            workspace_handler.d_missing_index_ + workspace_handler.h_shuffled_embedding_offset_[i];
        fill_default_emb_vec_async(d_vals_merge_dst_ptr,
                                   cache_config_.default_value_for_each_table[i],
                                   d_missing_index_ptr, workspace_handler.h_missing_length_[i],
                                   cache_config_.embedding_vec_size_[i], BLOCK_SIZE_, streams[i]);

        size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                              workspace_handler.h_shuffled_embedding_offset_[i];
        acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      }
    }

    // Decompress the hit emb_vec buffer to output buffer
    acc_emb_vec_offset = 0;
    size_t acc_emb_table_offset = 0;
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      float* d_unique_src_ptr = workspace_handler.d_hit_emb_vec_ + acc_emb_vec_offset;
      float* d_decompress_dst_ptr = d_shuffled_embeddingoutputvector + acc_emb_table_offset;
      uint64_t* d_output_index_ptr = workspace_handler.d_unique_output_index_ +
                                     workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                            workspace_handler.h_shuffled_embedding_offset_[i];
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      acc_emb_table_offset +=
          cache_config_.max_query_len_per_emb_table_[i] * cache_config_.embedding_vec_size_[i];
      decompress_emb_vec_async(d_unique_src_ptr, d_output_index_ptr, d_decompress_dst_ptr,
                               query_length, cache_config_.embedding_vec_size_[i], BLOCK_SIZE_,
                               streams[i]);
    }

    // Clear the unique op object to be ready for next look_up
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      ((unique_op_*)(workspace_handler.unique_op_obj_[i]))->clear(streams[i]);
    }
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
    }

    // Handle the missing keys, mode 2: synchronous
    if (async_insert_flag) {
      std::lock_guard<std::mutex> lock(mutex_);
      ps_insert_threads_.emplace_back(
          parameter_server_insert_thread_func_<TypeHashKey>, parameter_server_, this, memory_block,
          std::ref(cache_config_), std::ref(insert_streams_), std::ref(stream_mutex_));
    }
    return !async_insert_flag;
  } else {
    // Query the shuffled embeddingcolumns from Parameter Server & copy to device output buffer
    size_t acc_emb_vec_offset = 0;
    size_t acc_emb_table_offset = 0;
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      TypeHashKey* h_query_key_ptr =
          (TypeHashKey*)(workspace_handler.h_shuffled_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                            workspace_handler.h_shuffled_embedding_offset_[i];
      size_t query_length_in_float = query_length * cache_config_.embedding_vec_size_[i];
      size_t query_length_in_byte = query_length_in_float * sizeof(float);
      float* h_vals_retrieved_ptr = workspace_handler.h_missing_emb_vec_ + acc_emb_vec_offset;
      float* d_vals_retrieved_ptr = d_shuffled_embeddingoutputvector + acc_emb_table_offset;
      acc_emb_vec_offset += query_length_in_float;
      acc_emb_table_offset +=
          cache_config_.max_query_len_per_emb_table_[i] * cache_config_.embedding_vec_size_[i];
      parameter_server_->look_up(h_query_key_ptr, query_length, h_vals_retrieved_ptr,
                                 cache_config_.model_name_, i);
      CK_CUDA_THROW_(cudaMemcpyAsync(d_vals_retrieved_ptr, h_vals_retrieved_ptr,
                                     query_length_in_byte, cudaMemcpyHostToDevice, streams[i]));
    }
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      CK_CUDA_THROW_(cudaStreamSynchronize(streams[i]));
    }
    return true;
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::update(embedding_cache_workspace& workspace_handler,
                                          const std::vector<cudaStream_t>& streams) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing update
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    size_t acc_emb_vec_offset = 0;
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      TypeHashKey* d_missing_key_ptr =
          (TypeHashKey*)(workspace_handler.d_missing_embeddingcolumns_) +
          workspace_handler.h_shuffled_embedding_offset_[i];
      float* d_vals_retrieved_ptr = workspace_handler.d_missing_emb_vec_ + acc_emb_vec_offset;
      size_t query_length = workspace_handler.h_shuffled_embedding_offset_[i + 1] -
                            workspace_handler.h_shuffled_embedding_offset_[i];
      acc_emb_vec_offset += query_length * cache_config_.embedding_vec_size_[i];
      gpu_emb_caches_[i]->Replace(d_missing_key_ptr, workspace_handler.h_missing_length_[i],
                                  d_vals_retrieved_ptr, streams[i]);
    }
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::Dump(int table_id, void* key_buffer, size_t* length,
                                        size_t start_index, size_t end_index, cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Check for corner case
    if (start_index >= cache_config_.num_set_in_cache_[table_id]) {
      CK_THROW_(Error_t::WrongInput, "Error: Invalid value for start_index.");
    }
    if (end_index <= start_index || end_index > cache_config_.num_set_in_cache_[table_id]) {
      CK_THROW_(Error_t::WrongInput, "Error: Invalid value for end_index.");
    }
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing Dump
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    // Call GPU cache API
    gpu_emb_caches_[table_id]->Dump((TypeHashKey*)key_buffer, length, start_index, end_index,
                                    stream);
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::Refresh(int table_id, void* key_buffer, float* vec_buffer,
                                           size_t length, cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Check for corner case
    if (length == 0) {
      return;
    }
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before doing Refresh
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    // Call GPU cache API
    gpu_emb_caches_[table_id]->Update((TypeHashKey*)key_buffer, length, vec_buffer, stream,
                                      SLAB_SIZE);
  }
}

template <typename TypeHashKey>
embedding_cache_workspace embedding_cache<TypeHashKey>::create_workspace() {
  embedding_cache_workspace workspace_handler;
  size_t max_query_len_per_batch = 0;
  size_t max_emb_vec_len_per_batch_in_float = 0;
  for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
    max_query_len_per_batch += cache_config_.max_query_len_per_emb_table_[i];
    max_emb_vec_len_per_batch_in_float +=
        (cache_config_.max_query_len_per_emb_table_[i] * cache_config_.embedding_vec_size_[i]);
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
  workspace_handler.use_gpu_embedding_cache_ = cache_config_.use_gpu_embedding_cache_;
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before creating workspace buffer
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_shuffled_embeddingcolumns_,
                              max_query_len_per_batch * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_unique_output_index_,
                              max_query_len_per_batch * sizeof(uint64_t)));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_unique_output_embeddingcolumns_,
                              max_query_len_per_batch * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_unique_length_,
                              cache_config_.num_emb_table_ * sizeof(size_t)));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_unique_length_,
                                 cache_config_.num_emb_table_ * sizeof(size_t),
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaMalloc((void**)&workspace_handler.d_hit_emb_vec_,
                              max_emb_vec_len_per_batch_in_float * sizeof(float)));
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
    CK_CUDA_THROW_(cudaHostAlloc((void**)&workspace_handler.h_hit_rate_,
                                 cache_config_.num_emb_table_ * sizeof(double),
                                 cudaHostAllocPortable));
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      size_t capacity =
          (size_t)(cache_config_.max_query_len_per_emb_table_[i] / UNIQUE_OP_LOAD_FACTOR);
      workspace_handler.unique_op_obj_.push_back((void*)(new unique_op_(capacity)));
    }
  }
  return workspace_handler;
}

template <typename TypeHashKey>
embedding_cache_refreshspace embedding_cache<TypeHashKey>::create_refreshspace() {
  embedding_cache_refreshspace refreshspace;
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    int max_num_cache_set = *max_element(cache_config_.num_set_in_cache_.begin(),
                                         cache_config_.num_set_in_cache_.end());
    int max_embedding_size = *max_element(cache_config_.embedding_vec_size_.begin(),
                                          cache_config_.embedding_vec_size_.end());
    size_t max_num_keys_set = (SLAB_SIZE * SET_ASSOCIATIVITY) * max_num_cache_set;
    size_t max_num_key_in_buffer =
        cache_config_.cache_refresh_percentage_per_iteration * max_num_keys_set;
    cache_config_.num_set_in_refresh_workspace_ =
        max_num_key_in_buffer / (SLAB_SIZE * SET_ASSOCIATIVITY);
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&refreshspace.h_refresh_embeddingcolumns_,
                                 max_num_key_in_buffer * sizeof(TypeHashKey),
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(cudaHostAlloc((void**)&refreshspace.h_refresh_emb_vec_,
                                 max_num_key_in_buffer * max_embedding_size * sizeof(float),
                                 cudaHostAllocPortable));
    CK_CUDA_THROW_(
        cudaHostAlloc((void**)&refreshspace.h_length_, sizeof(size_t), cudaHostAllocPortable));

    CK_CUDA_THROW_(cudaMalloc((void**)&refreshspace.d_refresh_embeddingcolumns_,
                              max_num_key_in_buffer * sizeof(TypeHashKey)));
    CK_CUDA_THROW_(cudaMalloc((void**)&refreshspace.d_refresh_emb_vec_,
                              max_num_key_in_buffer * max_embedding_size * sizeof(float)));
    CK_CUDA_THROW_(cudaMalloc((void**)&refreshspace.d_length_, sizeof(size_t)));
  }
  return refreshspace;
}

template <typename TypeHashKey>
embedding_cache_config embedding_cache<TypeHashKey>::get_cache_config() {
  return cache_config_;
};

template <typename TypeHashKey>
std::vector<cudaStream_t>& embedding_cache<TypeHashKey>::get_refresh_streams() {
  return refresh_streams_;
};

template <typename TypeHashKey>
void* embedding_cache<TypeHashKey>::get_worker_space(const std::string& model_name, int device_id,
                                                     CACHE_SPACE_TYPE space_type) {
  return (parameter_server_->ApplyBuffer(model_name, device_id, space_type));
};

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::free_worker_space(void* p) {
  parameter_server_->FreeBuffer(p);
  return;
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::destroy_workspace(embedding_cache_workspace& workspace_handler) {
  // Free common buffer
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_shuffled_embeddingcolumns_));
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_shuffled_embedding_offset_));
  CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_emb_vec_));
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Device Restorer
    CudaDeviceContext dev_restorer;
    // Set CUDA device before free workspace buffer
    CK_CUDA_THROW_(cudaSetDevice(cache_config_.cuda_dev_id_));

    CK_CUDA_THROW_(cudaFree(workspace_handler.d_shuffled_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_unique_output_index_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_unique_output_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_unique_length_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_unique_length_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_hit_emb_vec_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_length_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_missing_length_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_index_));
    CK_CUDA_THROW_(cudaFree(workspace_handler.d_missing_emb_vec_));
    CK_CUDA_THROW_(cudaFreeHost(workspace_handler.h_hit_rate_));
    for (unsigned int i = 0; i < cache_config_.num_emb_table_; i++) {
      delete ((unique_op_*)(workspace_handler.unique_op_obj_[i]));
    }
  }
}

template <typename TypeHashKey>
void embedding_cache<TypeHashKey>::destroy_refreshspace(
    embedding_cache_refreshspace& refreshspace_handler) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CK_CUDA_THROW_(cudaFreeHost(refreshspace_handler.h_refresh_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFreeHost(refreshspace_handler.h_refresh_emb_vec_));
    CK_CUDA_THROW_(cudaFreeHost(refreshspace_handler.h_length_));

    CK_CUDA_THROW_(cudaFree(refreshspace_handler.d_refresh_embeddingcolumns_));
    CK_CUDA_THROW_(cudaFree(refreshspace_handler.d_refresh_emb_vec_));
    CK_CUDA_THROW_(cudaFree(refreshspace_handler.d_length_));
  }
}

template class embedding_cache<unsigned int>;
template class embedding_cache<long long>;
template static void parameter_server_insert_thread_func_<long long>(
    HugectrUtility<long long>*, embedding_interface*, MemoryBlock*, embedding_cache_config&,
    const std::vector<cudaStream_t>&, std::mutex&);
template static void parameter_server_insert_thread_func_<unsigned int>(
    HugectrUtility<unsigned int>*, embedding_interface*, MemoryBlock*, embedding_cache_config&,
    const std::vector<cudaStream_t>&, std::mutex&);
}  // namespace HugeCTR
