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
#include <utils.hpp> 
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
#include <inference/gpu_cache/nv_gpu_cache.hpp>
#include <inference/gpu_cache/unique_op.hpp>

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

  // Allocate a copy of workspace memory for a worker, should be called once by a worker
  virtual embedding_cache_workspace create_workspace();

  // Free a copy of workspace memory for a worker, should be called once by a worker
  virtual void destroy_workspace(embedding_cache_workspace& workspace_handler);

  // Query embeddingcolumns
  virtual void look_up(const void* h_embeddingcolumns, // The input emb_id buffer(before shuffle) on host
                       const std::vector<size_t>& h_embedding_offset, // The input offset on host, size = (# of samples * # of emb_table) + 1
                       float* d_shuffled_embeddingoutputvector, // The output buffer for emb_vec result on device
                       embedding_cache_workspace& workspace_handler, // The handler to the workspace buffers
                       const std::vector<cudaStream_t>& streams); // The CUDA stream to launch kernel to each emb_cache for each emb_table, size = # of emb_table(cache)

  // Update the embedding cache with missing embeddingcolumns from query API
  virtual void update(embedding_cache_workspace& workspace_handler, 
                      const std::vector<cudaStream_t>& streams);

 private:
  static const size_t BLOCK_SIZE_ = 64;
  
  // The GPU embedding cache type
  using cache_ = gpu_cache::gpu_cache<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(), SET_ASSOCIATIVITY, SLAB_SIZE>;
  // The GPU unique op type
  using unique_op_ = unique_op::unique_op<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(), std::numeric_limits<uint64_t>::max()>;

  // The back-end parameter server
  HugectrUtility<TypeHashKey>* parameter_server_;

  // The shared thread-safe embedding cache
  std::vector<cache_*> gpu_emb_caches_;

  // The cache configuration
  embedding_cache_config cache_config_;
  
};

}  // namespace HugeCTR
