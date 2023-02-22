/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <hps/embedding_cache.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/memory_pool.hpp>
#include <hps/static_table.hpp>
#include <io/filesystem.hpp>
#include <memory>
#include <mutex>
#include <thread>
#include <utils.hpp>

namespace HugeCTR {

template <typename TypeHashKey>
static void parameter_server_insert_thread_func_(
    const size_t table_id, HierParameterServerBase* const parameter_server,
    std::shared_ptr<EmbeddingCacheBase> embedding_cache, MemoryBlock* const memory_block,
    cudaStream_t stream, std::mutex& stream_mutex) {
  try {
    // TODO: Why do we lock the mutex so early?
    std::lock_guard<std::mutex> lock(stream_mutex);
    // Create sync events.
    cudaEvent_t insert_event;
    cudaEventCreate(&insert_event);
    // Insert data.
    parameter_server->insert_embedding_cache(table_id, embedding_cache, memory_block->worker_buffer,
                                             stream);
    cudaEventRecord(insert_event, stream);
    // Await sync events.
    cudaEventSynchronize(insert_event);
    // Cleanup.
    parameter_server->free_buffer(memory_block);
    cudaEventDestroy(insert_event);
  } catch (const std::runtime_error& rt_err) {
    parameter_server->free_buffer(memory_block);
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

std::shared_ptr<EmbeddingCacheBase> EmbeddingCacheBase::create(
    const InferenceParams& inference_params, const parameter_server_config& ps_config,
    HierParameterServerBase* const parameter_server) {
  if (inference_params.use_static_table) {
    if (inference_params.i64_input_key) {
      return std::make_shared<StaticTable<long long>>(inference_params, ps_config,
                                                      parameter_server);
    } else {
      return std::make_shared<StaticTable<unsigned int>>(inference_params, ps_config,
                                                         parameter_server);
    }
  } else {
    if (inference_params.i64_input_key) {
      return std::make_shared<EmbeddingCache<long long>>(inference_params, ps_config,
                                                         parameter_server);
    } else {
      return std::make_shared<EmbeddingCache<unsigned int>>(inference_params, ps_config,
                                                            parameter_server);
    }
  }
}

EmbeddingCacheBase::~EmbeddingCacheBase() = default;

template <typename TypeHashKey>
EmbeddingCache<TypeHashKey>::EmbeddingCache(const InferenceParams& inference_params,
                                            const parameter_server_config& ps_config,
                                            HierParameterServerBase* const parameter_server)
    : EmbeddingCacheBase(),
      parameter_server_(parameter_server),
      insert_workers_("EC insert",
                      std::max(static_cast<unsigned int>(inference_params.thread_pool_size),
                               std::thread::hardware_concurrency())) {
  auto b2s = [](const char val) { return val ? "True" : "False"; };
  HCTR_LOG(INFO, ROOT, "Model name: %s\n", inference_params.model_name.c_str());
  HCTR_LOG(INFO, ROOT, "Max batch size: %lu\n", inference_params.max_batchsize);
  HCTR_LOG(INFO, ROOT, "Number of embedding tables: %zu\n",
           inference_params.sparse_model_files.size());
  HCTR_LOG(INFO, ROOT, "Use GPU embedding cache: %s, cache size percentage: %f\n",
           b2s(inference_params.use_gpu_embedding_cache), inference_params.cache_size_percentage);
  HCTR_LOG(INFO, ROOT, "Use static table: %s\n", b2s(inference_params.use_static_table));
  HCTR_LOG(INFO, ROOT, "Use I64 input key: %s\n", b2s(inference_params.i64_input_key));
  HCTR_LOG(INFO, ROOT, "Configured cache hit rate threshold: %f\n",
           inference_params.hit_rate_threshold);
  HCTR_LOG(INFO, ROOT, "The size of thread pool: %u\n",
           std::max(static_cast<unsigned int>(inference_params.thread_pool_size),
                    std::thread::hardware_concurrency()));
  HCTR_LOG(INFO, ROOT, "The size of worker memory pool: %u\n",
           inference_params.number_of_worker_buffers_in_pool);
  HCTR_LOG(INFO, ROOT, "The size of refresh memory pool: %u\n",
           inference_params.number_of_refresh_buffers_in_pool);
  HCTR_LOG(INFO, ROOT, "The refresh percentage : %f\n",
           inference_params.cache_refresh_percentage_per_iteration);

  // initialize the profiler
  ec_profiler_ = std::make_unique<profiler>(ProfilerTarget_t::EC);

  // Store the configuration
  cache_config_.num_emb_table_ = inference_params.sparse_model_files.size();
  cache_config_.cache_size_percentage_ = inference_params.cache_size_percentage;
  cache_config_.cache_refresh_percentage_per_iteration =
      inference_params.cache_refresh_percentage_per_iteration;
  cache_config_.default_value_for_each_table = inference_params.default_value_for_each_table;
  cache_config_.model_name_ = inference_params.model_name;
  cache_config_.cuda_dev_id_ = inference_params.device_id;
  cache_config_.use_gpu_embedding_cache_ = inference_params.use_gpu_embedding_cache;

  if (ps_config.embedding_vec_size_.find(inference_params.model_name) ==
          ps_config.embedding_vec_size_.end() ||
      ps_config.emb_table_name_.find(inference_params.model_name) ==
          ps_config.emb_table_name_.end() ||
      ps_config.max_feature_num_per_sample_per_emb_table_.find(inference_params.model_name) ==
          ps_config.max_feature_num_per_sample_per_emb_table_.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "The model_name is not in the parameter server configurations");
  } else {
    cache_config_.embedding_vec_size_ =
        ps_config.embedding_vec_size_.at(inference_params.model_name);
    cache_config_.embedding_table_name_ = ps_config.emb_table_name_.at(inference_params.model_name);
    auto max_feature_num_per_sample =
        ps_config.max_feature_num_per_sample_per_emb_table_.at(inference_params.model_name);
    cache_config_.max_query_len_per_emb_table_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      cache_config_.max_query_len_per_emb_table_.emplace_back(inference_params.max_batchsize *
                                                              max_feature_num_per_sample[i]);
    }
  }

  // Query the size of all embedding tables and calculate the size of each embedding cache
  if (cache_config_.use_gpu_embedding_cache_) {
    cache_config_.num_set_in_cache_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      const size_t row_num = ps_config.embedding_key_count_.at(inference_params.model_name)[i];
      size_t num_feature_in_cache = static_cast<size_t>(
          static_cast<double>(cache_config_.cache_size_percentage_) * static_cast<double>(row_num));
      if (num_feature_in_cache < SLAB_SIZE * SET_ASSOCIATIVITY) {
        num_feature_in_cache = SLAB_SIZE * SET_ASSOCIATIVITY;
        HCTR_LOG(INFO, ROOT,
                 "The initial size of the embedding cache is smaller than the minimum setting: "
                 "\"%d\" and %d will be used as the default embedding cache size.\n",
                 num_feature_in_cache, SLAB_SIZE * SET_ASSOCIATIVITY);
      }
      cache_config_.num_set_in_cache_.emplace_back(
          (num_feature_in_cache + SLAB_SIZE * SET_ASSOCIATIVITY - 1) /
          (SLAB_SIZE * SET_ASSOCIATIVITY));
    }
  }

  // Construct gpu embedding cache, 1 per embedding table
  if (cache_config_.use_gpu_embedding_cache_) {
    // This is the only two places to set the cuda context in embedding cache
    CudaDeviceContext dev_restorer;
    dev_restorer.set_device(cache_config_.cuda_dev_id_);

    // Allocate resources.
    gpu_emb_caches_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      gpu_emb_caches_.emplace_back(std::make_unique<Cache>(cache_config_.num_set_in_cache_[i],
                                                           cache_config_.embedding_vec_size_[i]));
    }

    insert_streams_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      insert_streams_.push_back(stream);
    }

    refresh_streams_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      refresh_streams_.push_back(stream);
    }
  }
}

template <typename TypeHashKey>
EmbeddingCache<TypeHashKey>::~EmbeddingCache() {
  if (cache_config_.use_gpu_embedding_cache_) {
    // This is the only two places to set the cuda context in embedding cache
    CudaDeviceContext dev_restorer;
    dev_restorer.set_device(cache_config_.cuda_dev_id_);

    // Destroy resources.
    for (auto& stream : insert_streams_) {
      cudaStreamDestroy(stream);
    }
    insert_streams_.clear();
    for (auto& stream : refresh_streams_) {
      cudaStreamDestroy(stream);
    }
    refresh_streams_.clear();

    gpu_emb_caches_.clear();
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::lookup(size_t const table_id, float* const d_vectors,
                                         const void* const h_keys, size_t const num_keys,
                                         float const hit_rate_threshold, cudaStream_t stream) {
  MemoryBlock* memory_block = nullptr;
  BaseUnit* start = profiler::start();
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(parameter_server_->apply_buffer(
        cache_config_.model_name_, cache_config_.cuda_dev_id_, CACHE_SPACE_TYPE::WORKER));
  }
  ec_profiler_->end(start, "Apply for workspace from the memory pool for Embedding Cache Lookup");
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    // Copy the keys to device
    start = profiler::start();
    HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_embeddingcolumns_[table_id], h_keys,
                                   num_keys * sizeof(TypeHashKey), cudaMemcpyHostToDevice, stream));
    ec_profiler_->end(start, "Copy the input to workspace of Embedding Cache",
                      ProfilerType_t::Timeliness, stream);
    start = profiler::start();
    lookup_from_device(table_id, d_vectors, memory_block, num_keys, hit_rate_threshold, stream);
    ec_profiler_->end(start, "Lookup the embedding keys from Embedding Cache");
  }
  // Not using GPU embedding cache
  else {
    memcpy(workspace_handler.h_embeddingcolumns_[table_id], h_keys, num_keys * sizeof(TypeHashKey));
    start = profiler::start();
    parameter_server_->lookup(workspace_handler.h_embeddingcolumns_[table_id], num_keys,
                              workspace_handler.h_missing_emb_vec_[table_id],
                              cache_config_.model_name_, table_id);
    ec_profiler_->end(
        start, "Lookup the embedding keys from Database backend(disable the Embedding Cache)");
    HCTR_LIB_THROW(
        cudaMemcpyAsync(d_vectors, workspace_handler.h_missing_emb_vec_[table_id],
                        num_keys * cache_config_.embedding_vec_size_[table_id] * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    parameter_server_->free_buffer(memory_block);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::lookup_from_device(size_t const table_id, float* const d_vectors,
                                                     const void* const d_keys,
                                                     size_t const num_keys,
                                                     float const hit_rate_threshold,
                                                     cudaStream_t stream) {
  MemoryBlock* memory_block = nullptr;
  BaseUnit* start = profiler::start();
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(parameter_server_->apply_buffer(
        cache_config_.model_name_, cache_config_.cuda_dev_id_, CACHE_SPACE_TYPE::WORKER));
  }
  ec_profiler_->end(
      start, "Apply for workspace from the memory pool for Embedding Cache Lookup_from_device");
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;

  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_embeddingcolumns_[table_id], d_keys,
                                   num_keys * sizeof(TypeHashKey), cudaMemcpyDeviceToDevice,
                                   stream));
    start = profiler::start();
    lookup_from_device(table_id, d_vectors, memory_block, num_keys, hit_rate_threshold, stream);
    ec_profiler_->end(start, "Lookup the embedding keys from Embedding Cache");
  }
  // Not using GPU embedding cache
  else {
    HCTR_LIB_THROW(cudaMemcpy(workspace_handler.h_embeddingcolumns_[table_id], d_keys,
                              num_keys * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
    start = profiler::start();
    parameter_server_->lookup(workspace_handler.h_embeddingcolumns_[table_id], num_keys,
                              workspace_handler.h_missing_emb_vec_[table_id],
                              cache_config_.model_name_, table_id);
    ec_profiler_->end(
        start, "Lookup the embedding keys from Database backend(disable the Embedding Cache)");
    HCTR_LIB_THROW(
        cudaMemcpyAsync(d_vectors, workspace_handler.h_missing_emb_vec_[table_id],
                        num_keys * cache_config_.embedding_vec_size_[table_id] * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    parameter_server_->free_buffer(memory_block);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::lookup_from_device(size_t const table_id, float* const d_vectors,
                                                     MemoryBlock* memory_block,
                                                     size_t const num_keys,
                                                     float const hit_rate_threshold,
                                                     cudaStream_t stream) {
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    BaseUnit* start = profiler::start();
    // Unique
    static_cast<UniqueOp*>(workspace_handler.unique_op_obj_[table_id])
        ->unique(static_cast<TypeHashKey*>(workspace_handler.d_embeddingcolumns_[table_id]),
                 num_keys, workspace_handler.d_unique_output_index_[table_id],
                 static_cast<TypeHashKey*>(
                     workspace_handler.d_unique_output_embeddingcolumns_[table_id]),
                 workspace_handler.d_unique_length_ + table_id, stream);
    HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.h_unique_length_ + table_id,
                                   workspace_handler.d_unique_length_ + table_id, sizeof(size_t),
                                   cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    ec_profiler_->end(start, "Deduplicate the input embedding key for Embedding Cache");

    // Query
    const size_t query_length = workspace_handler.h_unique_length_[table_id];
    const size_t task_per_warp_tile = (query_length < 1000000) ? 1 : 32;
    start = profiler::start();
    gpu_emb_caches_[table_id]->Query(
        static_cast<TypeHashKey*>(workspace_handler.d_unique_output_embeddingcolumns_[table_id]),
        workspace_handler.h_unique_length_[table_id], workspace_handler.d_hit_emb_vec_[table_id],
        workspace_handler.d_missing_index_[table_id],
        static_cast<TypeHashKey*>(workspace_handler.d_missing_embeddingcolumns_[table_id]),
        workspace_handler.d_missing_length_ + table_id, stream, task_per_warp_tile);
    HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.h_missing_length_ + table_id,
                                   workspace_handler.d_missing_length_ + table_id, sizeof(size_t),
                                   cudaMemcpyDeviceToHost, stream));
    // Set async flag
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    ec_profiler_->end(start, "Native Embedding Cache Query API");
    if (workspace_handler.h_unique_length_[table_id] == 0) {
      workspace_handler.h_hit_rate_[table_id] = 1.0;
    } else {
      workspace_handler.h_hit_rate_[table_id] =
          1.0 - (static_cast<double>(workspace_handler.h_missing_length_[table_id]) /
                 static_cast<double>(workspace_handler.h_unique_length_[table_id]));
    }

    bool async_insert_flag{workspace_handler.h_hit_rate_[table_id] >= hit_rate_threshold};
    start = profiler::start(workspace_handler.h_hit_rate_[table_id], ProfilerType_t::Occupancy);
    ec_profiler_->end(start, "The hit rate of Embedding Cache", ProfilerType_t::Occupancy);
    // Handle the missing keys
    // mode 1: synchronous
    if (!async_insert_flag) {
      start = profiler::start();
      parameter_server_->insert_embedding_cache(table_id, this->shared_from_this(),
                                                workspace_handler, stream);
      // Wait for memory copy to complete
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      ec_profiler_->end(start, "Missing key synchronization insert into Embedding Cache");
      start = profiler::start();
      merge_emb_vec_async(workspace_handler.d_hit_emb_vec_[table_id],
                          workspace_handler.d_missing_emb_vec_[table_id],
                          workspace_handler.d_missing_index_[table_id],
                          workspace_handler.h_missing_length_[table_id],
                          cache_config_.embedding_vec_size_[table_id], BLOCK_SIZE_, stream);
      ec_profiler_->end(start, "Merge output from Embedding Cache", ProfilerType_t::Timeliness,
                        stream);
    }
    // mode 2: Asynchronous
    else {
      start = profiler::start();
      fill_default_emb_vec_async(workspace_handler.d_hit_emb_vec_[table_id],
                                 cache_config_.default_value_for_each_table[table_id],
                                 workspace_handler.d_missing_index_[table_id],
                                 workspace_handler.h_missing_length_[table_id],
                                 cache_config_.embedding_vec_size_[table_id], BLOCK_SIZE_, stream);
      ec_profiler_->end(start, "Fill default embedding vector asynchronously",
                        ProfilerType_t::Timeliness, stream);
    }
    start = profiler::start();
    // Decompress the hit emb_vec buffer to output buffer
    decompress_emb_vec_async(workspace_handler.d_hit_emb_vec_[table_id],
                             workspace_handler.d_unique_output_index_[table_id], d_vectors,
                             num_keys, cache_config_.embedding_vec_size_[table_id], BLOCK_SIZE_,
                             stream);
    // Clear the unique op object to be ready for next lookup
    static_cast<UniqueOp*>(workspace_handler.unique_op_obj_[table_id])->clear(stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    ec_profiler_->end(start, "decompress/deunique output from Embedding Cache");

    // Handle the missing keys, mode 2: synchronous
    if (async_insert_flag) {
      std::lock_guard<std::mutex> lock(mutex_);
      insert_workers_.submit([this, self(this->shared_from_this()), table_id, memory_block]() {
        parameter_server_insert_thread_func_<TypeHashKey>(table_id, parameter_server_, self,
                                                          memory_block, insert_streams_[table_id],
                                                          stream_mutex_);
      });
    } else {
      parameter_server_->free_buffer(memory_block);
    }
  } else {
    HCTR_LOG_S(ERROR, WORLD)
        << "Cannot call internal lookup_from_device when disabling GPU embedding cache"
        << std::endl;
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::insert(const size_t table_id,
                                         EmbeddingCacheWorkspace& workspace_handler,
                                         cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    gpu_emb_caches_[table_id]->Replace(
        static_cast<TypeHashKey*>(workspace_handler.d_missing_embeddingcolumns_[table_id]),
        workspace_handler.h_missing_length_[table_id],
        workspace_handler.d_missing_emb_vec_[table_id], stream);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::init(const size_t table_id,
                                       EmbeddingCacheRefreshspace& refreshspace_handler,
                                       cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    gpu_emb_caches_[table_id]->Replace(
        static_cast<TypeHashKey*>(refreshspace_handler.d_refresh_embeddingcolumns_),
        *refreshspace_handler.h_length_, refreshspace_handler.d_refresh_emb_vec_, stream);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::dump(const size_t table_id, void* const d_keys,
                                       size_t* const d_length, const size_t start_index,
                                       const size_t end_index, cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Check for corner case
    if (start_index >= cache_config_.num_set_in_cache_[table_id]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Error: Invalid value for start_index.");
    }
    if (end_index <= start_index || end_index > cache_config_.num_set_in_cache_[table_id]) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Error: Invalid value for end_index.");
    }
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    // Call GPU cache API
    BaseUnit* start = profiler::start();
    gpu_emb_caches_[table_id]->Dump(static_cast<TypeHashKey*>(d_keys), d_length, start_index,
                                    end_index, stream);
    ec_profiler_->end(start, "Dump the exist keys from Embedding Cache", ProfilerType_t::Timeliness,
                      stream);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::refresh(const size_t table_id, const void* const d_keys,
                                          const float* const d_vectors, const size_t length,
                                          cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // Check for corner case
    if (length == 0) {
      return;
    }
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    BaseUnit* start = profiler::start();
    // Call GPU cache API
    gpu_emb_caches_[table_id]->Update(static_cast<const TypeHashKey*>(d_keys), length, d_vectors,
                                      stream, SLAB_SIZE);
    ec_profiler_->end(start, "Refresh/Update exist embedding vector in Embedding cache",
                      ProfilerType_t::Timeliness, stream);
  }
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::finalize() {
  if (cache_config_.use_gpu_embedding_cache_) {
    std::lock_guard<std::mutex> lock(mutex_);

    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    // Join insert threads
    insert_workers_.await_idle();
  }
}

template <typename TypeHashKey>
EmbeddingCacheWorkspace EmbeddingCache<TypeHashKey>::create_workspace() {
  EmbeddingCacheWorkspace workspace_handler;
  // Allocate common buffer.
  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    void* h_embeddingcolumns;
    HCTR_LIB_THROW(cudaHostAlloc(
        &h_embeddingcolumns, cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey),
        cudaHostAllocPortable));
    workspace_handler.h_embeddingcolumns_.push_back(h_embeddingcolumns);

    float* h_missing_emb_vec;
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&h_missing_emb_vec),
                                 cache_config_.max_query_len_per_emb_table_[i] *
                                     cache_config_.embedding_vec_size_[i] * sizeof(float),
                                 cudaHostAllocPortable));
    workspace_handler.h_missing_emb_vec_.push_back(h_missing_emb_vec);
  }
  // If GPU embedding cache is enabled.
  workspace_handler.use_gpu_embedding_cache_ = cache_config_.use_gpu_embedding_cache_;
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      void* d_embeddingcolumns;
      HCTR_LIB_THROW(cudaMalloc(&d_embeddingcolumns, cache_config_.max_query_len_per_emb_table_[i] *
                                                         sizeof(TypeHashKey)));
      workspace_handler.d_embeddingcolumns_.push_back(d_embeddingcolumns);

      uint64_t* d_unique_output_index;
      HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_unique_output_index),
                                cache_config_.max_query_len_per_emb_table_[i] * sizeof(uint64_t)));
      workspace_handler.d_unique_output_index_.push_back(d_unique_output_index);

      void* d_unique_output_embeddingcolumns;
      HCTR_LIB_THROW(
          cudaMalloc(&d_unique_output_embeddingcolumns,
                     cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey)));
      workspace_handler.d_unique_output_embeddingcolumns_.push_back(
          d_unique_output_embeddingcolumns);

      float* d_hit_emb_vec;
      HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_hit_emb_vec),
                                cache_config_.max_query_len_per_emb_table_[i] *
                                    cache_config_.embedding_vec_size_[i] * sizeof(float)));
      workspace_handler.d_hit_emb_vec_.push_back(d_hit_emb_vec);

      void* d_missing_embeddingcolumns;
      HCTR_LIB_THROW(
          cudaMalloc(&d_missing_embeddingcolumns,
                     cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey)));
      workspace_handler.d_missing_embeddingcolumns_.push_back(d_missing_embeddingcolumns);

      void* h_missing_embeddingcolumns;
      HCTR_LIB_THROW(
          cudaHostAlloc(&h_missing_embeddingcolumns,
                        cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey),
                        cudaHostAllocPortable));
      workspace_handler.h_missing_embeddingcolumns_.push_back(h_missing_embeddingcolumns);

      uint64_t* d_missing_index;
      HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_missing_index),
                                cache_config_.max_query_len_per_emb_table_[i] * sizeof(uint64_t)));
      workspace_handler.d_missing_index_.push_back(d_missing_index);

      float* d_missing_emb_vec;
      HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_missing_emb_vec),
                                cache_config_.max_query_len_per_emb_table_[i] *
                                    cache_config_.embedding_vec_size_[i] * sizeof(float)));
      workspace_handler.d_missing_emb_vec_.push_back(d_missing_emb_vec);

      const size_t capacity = static_cast<size_t>(cache_config_.max_query_len_per_emb_table_[i] /
                                                  UNIQUE_OP_LOAD_FACTOR);
      workspace_handler.unique_op_obj_.push_back(new UniqueOp(capacity));
    }
    HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&workspace_handler.d_missing_length_),
                              cache_config_.num_emb_table_ * sizeof(size_t)));
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&workspace_handler.h_missing_length_),
                                 cache_config_.num_emb_table_ * sizeof(size_t),
                                 cudaHostAllocPortable));
    HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&workspace_handler.d_unique_length_),
                              cache_config_.num_emb_table_ * sizeof(size_t)));
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&workspace_handler.h_unique_length_),
                                 cache_config_.num_emb_table_ * sizeof(size_t),
                                 cudaHostAllocPortable));
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&workspace_handler.h_hit_rate_),
                                 cache_config_.num_emb_table_ * sizeof(double),
                                 cudaHostAllocPortable));
  }
  return workspace_handler;
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::destroy_workspace(EmbeddingCacheWorkspace& workspace_handler) {
  // Free common buffer.
  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_embeddingcolumns_[i]));
    workspace_handler.h_embeddingcolumns_[i] = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_missing_emb_vec_[i]));
    workspace_handler.h_missing_emb_vec_[i] = nullptr;
  }
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      // Release resources.
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_embeddingcolumns_[i]));
      workspace_handler.d_embeddingcolumns_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_unique_output_index_[i]));
      workspace_handler.d_unique_output_index_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_unique_output_embeddingcolumns_[i]));
      workspace_handler.d_unique_output_embeddingcolumns_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_hit_emb_vec_[i]));
      workspace_handler.d_hit_emb_vec_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_missing_embeddingcolumns_[i]));
      workspace_handler.d_missing_embeddingcolumns_[i] = nullptr;
      HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_missing_embeddingcolumns_[i]));
      workspace_handler.h_missing_embeddingcolumns_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_missing_index_[i]));
      workspace_handler.d_missing_index_[i] = nullptr;
      HCTR_LIB_THROW(cudaFree(workspace_handler.d_missing_emb_vec_[i]));
      workspace_handler.d_missing_emb_vec_[i] = nullptr;
      delete static_cast<UniqueOp*>(workspace_handler.unique_op_obj_[i]);
    }

    HCTR_LIB_THROW(cudaFree(workspace_handler.d_unique_length_));
    workspace_handler.d_unique_length_ = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_unique_length_));
    workspace_handler.h_unique_length_ = nullptr;
    HCTR_LIB_THROW(cudaFree(workspace_handler.d_missing_length_));
    workspace_handler.d_missing_length_ = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_missing_length_));
    workspace_handler.h_missing_length_ = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_hit_rate_));
    workspace_handler.h_hit_rate_ = nullptr;
  }
}

template <typename TypeHashKey>
EmbeddingCacheRefreshspace EmbeddingCache<TypeHashKey>::create_refreshspace() {
  EmbeddingCacheRefreshspace refreshspace_handler;
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    const int max_num_cache_set = *max_element(cache_config_.num_set_in_cache_.begin(),
                                               cache_config_.num_set_in_cache_.end());
    const int max_embedding_size = *max_element(cache_config_.embedding_vec_size_.begin(),
                                                cache_config_.embedding_vec_size_.end());
    const size_t max_num_keys = (SLAB_SIZE * SET_ASSOCIATIVITY) * max_num_cache_set;
    const size_t max_num_key_in_buffer =
        std::max(float(SLAB_SIZE * SET_ASSOCIATIVITY),
                 cache_config_.cache_refresh_percentage_per_iteration* max_num_keys);
    cache_config_.num_set_in_refresh_workspace_ =
        (max_num_key_in_buffer + SLAB_SIZE * SET_ASSOCIATIVITY - 1) /
        (SLAB_SIZE * SET_ASSOCIATIVITY);

    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    // Create memory buffers.
    HCTR_LIB_THROW(cudaHostAlloc(&refreshspace_handler.h_refresh_embeddingcolumns_,
                                 max_num_key_in_buffer * sizeof(TypeHashKey),
                                 cudaHostAllocPortable));
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&refreshspace_handler.h_refresh_emb_vec_),
                                 max_num_key_in_buffer * max_embedding_size * sizeof(float),
                                 cudaHostAllocPortable));
    HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&refreshspace_handler.h_length_),
                                 sizeof(size_t), cudaHostAllocPortable));
    HCTR_LIB_THROW(cudaMalloc(&refreshspace_handler.d_refresh_embeddingcolumns_,
                              max_num_key_in_buffer * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&refreshspace_handler.d_refresh_emb_vec_),
                              max_num_key_in_buffer * max_embedding_size * sizeof(float)));
    HCTR_LIB_THROW(
        cudaMalloc(reinterpret_cast<void**>(&refreshspace_handler.d_length_), sizeof(size_t)));
  }
  return refreshspace_handler;
}

template <typename TypeHashKey>
void EmbeddingCache<TypeHashKey>::destroy_refreshspace(
    EmbeddingCacheRefreshspace& refreshspace_handler) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    // Release resources.
    HCTR_LIB_THROW(cudaFreeHost(refreshspace_handler.h_refresh_embeddingcolumns_));
    refreshspace_handler.h_refresh_embeddingcolumns_ = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(refreshspace_handler.h_refresh_emb_vec_));
    refreshspace_handler.h_refresh_emb_vec_ = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(refreshspace_handler.h_length_));
    refreshspace_handler.h_length_ = nullptr;

    HCTR_LIB_THROW(cudaFree(refreshspace_handler.d_refresh_embeddingcolumns_));
    refreshspace_handler.d_refresh_embeddingcolumns_ = nullptr;
    HCTR_LIB_THROW(cudaFree(refreshspace_handler.d_refresh_emb_vec_));
    refreshspace_handler.d_refresh_emb_vec_ = nullptr;
    HCTR_LIB_THROW(cudaFree(refreshspace_handler.d_length_));
    refreshspace_handler.d_length_ = nullptr;
  }
}

template class EmbeddingCache<long long>;
template class EmbeddingCache<unsigned int>;

}  // namespace HugeCTR