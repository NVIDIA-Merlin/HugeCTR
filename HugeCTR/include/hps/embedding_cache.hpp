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
#include <cuda_runtime_api.h>

#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <hps/memory_pool.hpp>
#include <hps/unique_op/unique_op.hpp>
#include <memory>
#include <nv_gpu_cache.hpp>
#include <thread_pool.hpp>
namespace HugeCTR {

class MemoryBlock;

template <typename TypeHashKey>
class EmbeddingCache : public EmbeddingCacheBase,
                       public std::enable_shared_from_this<EmbeddingCache<TypeHashKey>> {
 public:
  virtual ~EmbeddingCache();
  EmbeddingCache(const InferenceParams& inference_params, const parameter_server_config& ps_config,
                 HierParameterServerBase* const parameter_server);
  EmbeddingCache(EmbeddingCache const&) = delete;
  EmbeddingCache& operator=(EmbeddingCache const&) = delete;

  virtual void lookup(size_t table_id, float* d_vectors, const void* h_keys, size_t num_keys,
                      float hit_rate_threshold, cudaStream_t stream);
  virtual void lookup_from_device(size_t table_id, float* d_vectors, const void* d_keys,
                                  size_t num_keys, float hit_rate_threshold, cudaStream_t stream);
  virtual void lookup_from_device(size_t table_id, float* d_vectors, MemoryBlock* memory_block,
                                  size_t num_keys, float hit_rate_threshold, cudaStream_t stream);
  virtual void insert(size_t table_id, EmbeddingCacheWorkspace& workspace_handler,
                      cudaStream_t stream);

  virtual void init(const size_t table_id, EmbeddingCacheRefreshspace& refreshspace_handler,
                    cudaStream_t stream);
  virtual void dump(size_t table_id, void* d_keys, size_t* d_length, size_t start_index,
                    size_t end_index, cudaStream_t stream);
  virtual void refresh(size_t table_id, const void* d_keys, const float* d_vectors, size_t length,
                       cudaStream_t stream);
  virtual void finalize();

  virtual EmbeddingCacheWorkspace create_workspace();
  virtual void destroy_workspace(EmbeddingCacheWorkspace&);
  virtual EmbeddingCacheRefreshspace create_refreshspace();
  virtual void destroy_refreshspace(EmbeddingCacheRefreshspace&);

  virtual const embedding_cache_config& get_cache_config() { return cache_config_; }
  virtual const std::vector<cudaStream_t>& get_refresh_streams() { return refresh_streams_; }
  virtual const std::vector<cudaStream_t>& get_insert_streams() { return insert_streams_; }
  virtual int get_device_id() { return cache_config_.cuda_dev_id_; }
  virtual bool use_gpu_embedding_cache() { return cache_config_.use_gpu_embedding_cache_; }
  virtual void set_profiler(int interation, int warmup, bool enable_bench) {
    ec_profiler_->set_config(interation, warmup, enable_bench);
  };
  virtual void profiler_print() { ec_profiler_->print(); };

 private:
  static const size_t BLOCK_SIZE_ = 64;

  using Cache = gpu_cache::gpu_cache<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(),
                                     SET_ASSOCIATIVITY, SLAB_SIZE>;
  using UniqueOp =
      unique_op::unique_op<TypeHashKey, uint64_t, std::numeric_limits<TypeHashKey>::max(),
                           std::numeric_limits<uint64_t>::max()>;

  // The parameter server that it is bound to
  HierParameterServerBase* parameter_server_;

  // The cache configuration
  embedding_cache_config cache_config_;

  // The shared thread-safe embedding cache
  std::vector<std::unique_ptr<Cache>> gpu_emb_caches_;

  // streams for asynchronous parameter server insert threads
  std::vector<cudaStream_t> insert_streams_;

  // streams for asynchronous parameter server refresh threads
  std::vector<cudaStream_t> refresh_streams_;

  // parameter server insert threads
  ThreadPool insert_workers_;

  // mutex for insert_threads_
  std::mutex mutex_;

  // mutex for insert_streams_
  std::mutex stream_mutex_;

  // benchmark profiler
  std::unique_ptr<profiler> ec_profiler_;
};

}  // namespace HugeCTR