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

#pragma once
#include <embedding.hpp>
#include <inference/embedding_interface.hpp>
#include <inference/memory_pool.hpp>
#include <inference/unique_op/unique_op.hpp>
#include <iostream>
#include <metrics.hpp>
#include <network.hpp>
#include <nv_gpu_cache.hpp>
#include <parser.hpp>
#include <string>
#include <thread>
#include <utility>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

template <typename TypeHashKey>
class embedding_cache : public embedding_interface {
 public:
  embedding_cache(const std::string& model_config_path, const InferenceParams& inference_params,
                  HugectrUtility<TypeHashKey>* parameter_server);

  virtual ~embedding_cache();

  // Allocate a copy of workspace memory for a worker, should be called once by a worker
  virtual embedding_cache_workspace create_workspace();

  // Allocate a copy of refresh space memory for embedding cache versionupdate,
  virtual embedding_cache_refreshspace create_refreshspace();

  // Free a copy of workspace memory for a worker, should be called once by a worker
  virtual void destroy_workspace(embedding_cache_workspace& workspace_handler);

  // Free a copy of refresh space for a worker, should be called once by a worker
  virtual void destroy_refreshspace(embedding_cache_refreshspace& refreshspace_handler);

  virtual embedding_cache_config get_cache_config();

  virtual std::vector<cudaStream_t>& get_refresh_streams();

  virtual void* get_worker_space(const std::string& model_name, int device_id,
                                 CACHE_SPACE_TYPE space_type = CACHE_SPACE_TYPE::WORKER);

  virtual void free_worker_space(void* p);

  // Query embeddingcolumns
  virtual bool look_up(
      const void* h_embeddingcolumns,  // The input emb_id buffer(before shuffle) on host
      const std::vector<size_t>& h_embedding_offset,  // The input offset on host, size = (# of
                                                      // samples * # of emb_table) + 1
      float* d_shuffled_embeddingoutputvector,  // The output buffer for emb_vec result on device
      MemoryBlock* memory_block,  // The memory block for the handler to the workspace buffers
      const std::vector<cudaStream_t>&
          streams,  // The CUDA stream to launch kernel to each emb_cache for each emb_table, size =
                    // # of emb_table(cache)
      float hit_rate_threshold = 1.0);

  // Update the embedding cache with missing embeddingcolumns from query API
  virtual void update(embedding_cache_workspace& workspace_handler,
                      const std::vector<cudaStream_t>& streams);

  virtual void Dump(int table_id, void* key_buffer, size_t* length, size_t start_index,
                    size_t end_index, cudaStream_t stream);

  virtual void Refresh(int table_id, void* key_buffer, float* vec_buffer, size_t length,
                       cudaStream_t stream);

  virtual void finalize();

  virtual int get_device_id() { return cache_config_.cuda_dev_id_; }

  virtual bool use_gpu_embedding_cache() { return cache_config_.use_gpu_embedding_cache_; }

 private:
  static const size_t BLOCK_SIZE_ = 64;

  // The GPU embedding cache type
  using cache_ =
      gpu_cache::gpu_cache<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(),
                           SET_ASSOCIATIVITY, SLAB_SIZE>;
  // The GPU unique op type
  using unique_op_ =
      unique_op::unique_op<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(),
                           std::numeric_limits<uint64_t>::max()>;

  // The back-end parameter server
  HugectrUtility<TypeHashKey>* parameter_server_;

  // The shared thread-safe embedding cache
  std::vector<cache_*> gpu_emb_caches_;

  // The cache configuration
  embedding_cache_config cache_config_;

  // parameter server insert threads
  std::vector<std::thread> ps_insert_threads_;

  // streams for asynchronous parameter server insert threads
  std::vector<cudaStream_t> insert_streams_;

  // streams for asynchronous parameter server refresh threads
  std::vector<cudaStream_t> refresh_streams_;

  // mutex for insert_threads_
  std::mutex mutex_;

  // mutex for insert_streams_
  std::mutex stream_mutex_;
};

}  // namespace HugeCTR
