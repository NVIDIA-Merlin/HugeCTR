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
#include <common.hpp>
#include <iostream>
#include <embedding.hpp>
#include <metrics.hpp>
#include <network.hpp>
#include <parser.hpp>
#include <utils.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <inference/embedding_interface.hpp>
#include <inference/inference_utils.hpp>

namespace HugeCTR {

template <typename TypeHashKey>
class embedding_cache : public embedding_interface {
 public:
  embedding_cache(HugectrUtility<TypeHashKey>* parameter_server, // The backend PS
                  int cuda_dev_id, // Which CUDA device this cache belongs to
                  bool use_gpu_embedding_cache, // Whether enable GPU embedding cache or not
                  // The ratio of (size of GPU embedding cache : size of embedding table) for all embedding table in this model. Should between (0.0, 1.0].
                  float cache_size_percentage, 
                  const std::string& model_config_path, 
                  const std::string& model_name);
  virtual ~embedding_cache();

  virtual void look_up(const void* h_embeddingcolumns, // The input emb_id buffer(before shuffle) on host
                       const std::vector<size_t>& h_embedding_offset, // The input offset on host, size = (# of samples * # of emb_table) + 1
                       void* d_shuffled_embeddingcolumns, // The shuffled emb_id buffer on device, same size as h_embeddingcolumns
                       void* h_shuffled_embeddingcolumns, // The shuffled emb_id buffer on host, same size as h_embeddingcolumns
                       std::vector<size_t>& h_shuffled_embedding_offset, // The offset of each emb_table in shuffled emb_id buffer on host, size = # of emb_table + 1
                       void* d_missing_embeddingcolumns, // The buffer to hold missing emb_id for each emb_table on device, same size as h_embeddingcolumns
                       void* h_missing_embeddingcolumns, // The buffer to hold missing emb_id for each emb_table on host, same size as h_embeddingcolumns
                       size_t* d_missing_length, // The buffer to hold missing length for each emb_table on device, size = # of emb_table
                       size_t* h_missing_length, // The buffer to hold missing length for each emb_table on host, size = # of emb_table
                       uint64_t* d_missing_index, // The buffer to hold missing index for each emb_table on device, same size as h_embeddingcolumns
                       float* d_missing_emb_vec, // The buffer to hold retrieved missing emb_vec on device, same size as d_shuffled_embeddingoutputvector
                       float* h_missing_emb_vec, // The buffer to hold retrieved missing emb_vec from PS on host, same size as d_shuffled_embeddingoutputvector
                       float* d_shuffled_embeddingoutputvector, // The output buffer for emb_vec result on device
                       const std::vector<cudaStream_t>& streams); // The CUDA stream to launch kernel to each embd_cache for each emb_table, size = # of emb_table(cache)

  virtual void update(const std::vector<size_t>& h_shuffled_embedding_offset, // The same buffer as look_up
                      const size_t* h_missing_length,
                      const void* d_missing_embeddingcolumns,
                      const float* d_missing_emb_vec,
                      const std::vector<cudaStream_t>& streams);

 private:
  // The back-end parameter server
  HugectrUtility<TypeHashKey>* parameter_server_;

  // The shared thread-safe embedding cache
  // Will be introduced later

  // The cache configuration
  embedding_cache_config cache_config_;
  
};

}  // namespace HugeCTR