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
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <common.hpp>
#include <inference/inference_utils.hpp>

namespace HugeCTR {

struct embedding_cache_workspace{
  void* d_shuffled_embeddingcolumns_; // The shuffled emb_id buffer on device, same size as h_embeddingcolumns
  void* h_shuffled_embeddingcolumns_; // The shuffled emb_id buffer on host, same size as h_embeddingcolumns
  size_t* h_shuffled_embedding_offset_; // The offset of each emb_table in shuffled emb_id buffer on host, size = # of emb_table + 1
  void* d_missing_embeddingcolumns_; // The buffer to hold missing emb_id for each emb_table on device, same size as h_embeddingcolumns
  void* h_missing_embeddingcolumns_; // The buffer to hold missing emb_id for each emb_table on host, same size as h_embeddingcolumns
  size_t* d_missing_length_; // The buffer to hold missing length for each emb_table on device, size = # of emb_table
  size_t* h_missing_length_; // The buffer to hold missing length for each emb_table on host, size = # of emb_table
  uint64_t* d_missing_index_; // The buffer to hold missing index for each emb_table on device, same size as h_embeddingcolumns
  float* d_missing_emb_vec_; // The buffer to hold retrieved missing emb_vec on device, same size as d_shuffled_embeddingoutputvector
  float* h_missing_emb_vec_; // The buffer to hold retrieved missing emb_vec from PS on host, same size as d_shuffled_embeddingoutputvector
};

struct embedding_cache_config{
  float cache_size_percentage_;
  std::string model_name_; // Which model this cache belongs to
  int cuda_dev_id_; // Which CUDA device this cache belongs to
  bool use_gpu_embedding_cache_; // Whether enable GPU embedding cache or not
  // Each vector will have the size of E(# of embedding tables in the model)
  size_t num_emb_table_; // # of embedding table in this model
  std::vector<size_t> embedding_vec_size_; // # of float in emb_vec
  std::vector<size_t> num_set_in_cache_; // # of cache set in the cache
  std::vector<size_t> max_query_len_per_emb_table_; // The max # of embeddingcolumns each inference instance(batch) will query from a embedding table
};

// Base interface class for embedding cache
// 1 instance per model per GPU
class embedding_interface{
 public:
  embedding_interface();
  virtual ~embedding_interface();

  // Allocate a copy of workspace memory for a worker, should be called once by a worker
  virtual void create_workspace(embedding_cache_workspace& workspace_handler) = 0;

  // Free a copy of workspace memory for a worker, should be called once by a worker
  virtual void destroy_workspace(embedding_cache_workspace& workspace_handler) = 0;

  // Query embeddingcolumns
  virtual void look_up(const void* h_embeddingcolumns, // The input emb_id buffer(before shuffle) on host
                       const std::vector<size_t>& h_embedding_offset, // The input offset on host, size = (# of samples * # of emb_table) + 1
                       float* d_shuffled_embeddingoutputvector, // The output buffer for emb_vec result on device
                       embedding_cache_workspace& workspace_handler, // The handler to the workspace buffers
                       const std::vector<cudaStream_t>& streams) = 0; // The CUDA stream to launch kernel to each emb_cache for each emb_table, size = # of emb_table(cache)

  // Update the embedding cache with missing embeddingcolumns from query API
  virtual void update(embedding_cache_workspace& workspace_handler, 
                      const std::vector<cudaStream_t>& streams) = 0;

  template <typename TypeHashKey>
  static embedding_interface* Create_Embedding_Cache(HugectrUtility<TypeHashKey>* parameter_server, // The backend PS
                  int cuda_dev_id, // Which CUDA device this cache belongs to
                  bool use_gpu_embedding_cache, // Whether enable GPU embedding cache or not
                  // The ratio of (size of GPU embedding cache : size of embedding table) for all embedding table in this model. Should between (0.0, 1.0].
                  float cache_size_percentage,
                  const std::string& model_config_path,
                  const std::string& model_name);

};

}  // namespace HugeCTR

