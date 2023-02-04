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

#include <hps/inference_utils.hpp>

namespace HugeCTR {

class HierParameterServerBase;

class EmbeddingCacheBase {
 public:
  virtual ~EmbeddingCacheBase() = 0;
  EmbeddingCacheBase() = default;
  EmbeddingCacheBase(EmbeddingCacheBase const&) = delete;
  EmbeddingCacheBase& operator=(EmbeddingCacheBase const&) = delete;

  static std::shared_ptr<EmbeddingCacheBase> create(
      const InferenceParams& inference_params, const parameter_server_config& ps_config,
      HierParameterServerBase* const parameter_server);

  virtual void lookup(size_t table_id, float* d_vectors, const void* h_keys, size_t num_keys,
                      float hit_rate_threshold, cudaStream_t stream) = 0;

  virtual void lookup_from_device(size_t table_id, float* d_vectors, const void* d_keys,
                                  size_t num_keys, float hit_rate_threshold,
                                  cudaStream_t stream) = 0;

  virtual void insert(size_t table_id, EmbeddingCacheWorkspace& workspace_handler,
                      cudaStream_t stream) = 0;
  virtual void init(const size_t table_id, EmbeddingCacheRefreshspace& refreshspace_handler,
                    cudaStream_t stream) = 0;
  virtual void dump(size_t table_id, void* d_keys, size_t* d_length, size_t start_index,
                    size_t end_index, cudaStream_t stream) = 0;
  virtual void refresh(size_t table_id, const void* d_keys, const float* d_vectors, size_t length,
                       cudaStream_t stream) = 0;
  virtual void finalize() = 0;

  virtual EmbeddingCacheWorkspace create_workspace() = 0;
  virtual void destroy_workspace(EmbeddingCacheWorkspace&) = 0;
  virtual EmbeddingCacheRefreshspace create_refreshspace() = 0;
  virtual void destroy_refreshspace(EmbeddingCacheRefreshspace&) = 0;

  virtual const embedding_cache_config& get_cache_config() = 0;
  virtual const std::vector<cudaStream_t>& get_refresh_streams() = 0;
  virtual const std::vector<cudaStream_t>& get_insert_streams() = 0;
  virtual int get_device_id() = 0;
  virtual bool use_gpu_embedding_cache() = 0;
  virtual void set_profiler(int interation, int warmup, bool enable_bench) = 0;
  virtual void profiler_print() = 0;
};

}  // namespace HugeCTR