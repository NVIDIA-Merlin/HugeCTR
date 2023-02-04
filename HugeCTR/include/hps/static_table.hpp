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
#include <static_table.hpp>
#include <thread_pool.hpp>
namespace HugeCTR {

class MemoryBlock;

template <typename TypeHashKey>
class StaticTable : public EmbeddingCacheBase,
                    public std::enable_shared_from_this<StaticTable<TypeHashKey>> {
 public:
  virtual ~StaticTable();
  StaticTable(const InferenceParams& inference_params, const parameter_server_config& ps_config,
              HierParameterServerBase* const parameter_server);
  StaticTable(StaticTable const&) = delete;
  StaticTable& operator=(StaticTable const&) = delete;

  virtual void lookup(size_t table_id, float* d_vectors, const void* h_keys, size_t num_keys,
                      float hit_rate_threshold, cudaStream_t stream) override;
  virtual void lookup_from_device(size_t table_id, float* d_vectors, const void* d_keys,
                                  size_t num_keys, float hit_rate_threshold,
                                  cudaStream_t stream) override;
  virtual void init(const size_t table_id, EmbeddingCacheRefreshspace& refreshspace_handler,
                    cudaStream_t stream) override;
  virtual void refresh(size_t table_id, const void* d_keys, const float* d_vectors, size_t length,
                       cudaStream_t stream) override;

  virtual EmbeddingCacheWorkspace create_workspace() override;
  virtual void destroy_workspace(EmbeddingCacheWorkspace&) override;
  virtual EmbeddingCacheRefreshspace create_refreshspace() override;
  virtual void destroy_refreshspace(EmbeddingCacheRefreshspace&) override;

  virtual const embedding_cache_config& get_cache_config() override { return cache_config_; }
  virtual const std::vector<cudaStream_t>& get_refresh_streams() override {
    return refresh_streams_;
  }
  virtual int get_device_id() override { return cache_config_.cuda_dev_id_; }
  virtual bool use_gpu_embedding_cache() override { return cache_config_.use_gpu_embedding_cache_; }
  virtual void profiler_print() { ec_profiler_->print(); };
  virtual void set_profiler(int interation, int warmup, bool enable_bench) {
    ec_profiler_->set_config(interation, warmup, enable_bench);
  };

 private:
  using Cache = gpu_cache::static_table<TypeHashKey>;

  // This function is not used for static table
  virtual const std::vector<cudaStream_t>& get_insert_streams() { return refresh_streams_; }
  virtual void insert(size_t table_id, EmbeddingCacheWorkspace& workspace_handler,
                      cudaStream_t stream) override{};
  virtual void finalize() override{};
  virtual void lookup_from_device(size_t table_id, float* d_vectors, MemoryBlock* memory_block,
                                  size_t num_keys, cudaStream_t stream);
  virtual void dump(size_t table_id, void* d_keys, size_t* d_length, size_t start_index,
                    size_t end_index, cudaStream_t stream) override{};

  // The parameter server that it is bound to
  HierParameterServerBase* parameter_server_;

  // streams for asynchronous parameter server refresh threads
  std::vector<cudaStream_t> refresh_streams_;

  // embedding tables
  std::vector<std::unique_ptr<Cache>> static_tables_;

  // The cache configuration
  embedding_cache_config cache_config_;

  // benchmark profiler
  std::unique_ptr<profiler> ec_profiler_;
};

}  // namespace HugeCTR