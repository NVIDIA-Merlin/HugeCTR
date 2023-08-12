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
#include <hps/embedding_cache_stoch.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/memory_pool.hpp>
#include <hps/static_table.hpp>
#include <io/filesystem.hpp>
#include <memory>
#include <mutex>
#include <thread>
#include <utils.hpp>

// Providing a seed value

namespace HugeCTR {

template <typename TypeHashKey>
static void parameter_server_insert_thread_func_2_(
    const size_t table_id, HierParameterServerBase* const parameter_server,
    std::shared_ptr<EmbeddingCacheStoch<TypeHashKey>> embedding_cache,
    MemoryBlock* const memory_block, cudaStream_t stream, std::mutex& stream_mutex) {
  try {
    EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
    typename EmbeddingCacheStoch<TypeHashKey>::WorkspacePrivateData* private_data =
        reinterpret_cast<typename EmbeddingCacheStoch<TypeHashKey>::WorkspacePrivateData*>(
            workspace_handler.private_data_[table_id]);

    // check if heuristic regarding hit rate require a replacement or we are in steady state
    // future use the data from the histogram to have more precise histogram: but right now we only
    // get the misses so data from histogram is very skewed

    std::uniform_real_distribution<float> dist;
    float hist = dist(private_data->rd_);
    if (hist > workspace_handler.h_hit_rate_[table_id]) {
      embedding_cache->Replace(
          table_id,
          static_cast<TypeHashKey*>(workspace_handler.h_missing_embeddingcolumns_[table_id]),
          workspace_handler.h_missing_length_[table_id],
          workspace_handler.d_missing_emb_vec_[table_id], workspace_handler.h_hit_rate_[table_id],
          private_data->modify_handle_, private_data->modify_event_, stream);
    }

    parameter_server->free_buffer(memory_block);
  } catch (const std::runtime_error& rt_err) {
    parameter_server->free_buffer(memory_block);
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

template <typename TypeHashKey>
EmbeddingCacheStoch<TypeHashKey>::EmbeddingCacheStoch(
    const InferenceParams& inference_params, const parameter_server_config& ps_config,
    HierParameterServerBase* const parameter_server)
    : EmbeddingCacheBase(),
      parameter_server_(parameter_server),
      insert_workers_("EC insert",
                      std::min(static_cast<unsigned int>(inference_params.thread_pool_size),
                               std::thread::hardware_concurrency())) {
  auto b2s = [](const char val) { return val ? "True" : "False"; };
  HCTR_LOG(INFO, ROOT, "Model name: %s\n", inference_params.model_name.c_str());
  HCTR_LOG(INFO, ROOT, "Max batch size: %lu\n", inference_params.max_batchsize);
  HCTR_LOG(INFO, ROOT, "Number of embedding tables: %zu\n",
           inference_params.sparse_model_files.size());
  HCTR_LOG(
      INFO, ROOT, "Use GPU embedding cache: %s, nv_implementation: %s, cache size percentage: %f\n",
      b2s(inference_params.use_gpu_embedding_cache),
      b2s(inference_params.use_hctr_cache_implementation), inference_params.cache_size_percentage);
  HCTR_LOG(INFO, ROOT, "Use static table: %s\n", b2s(inference_params.use_static_table));
  HCTR_LOG(INFO, ROOT, "Use I64 input key: %s\n", b2s(inference_params.i64_input_key));
  HCTR_LOG(INFO, ROOT, "Configured cache hit rate threshold: %f\n",
           inference_params.hit_rate_threshold);
  HCTR_LOG(INFO, ROOT, "The size of thread pool: %u\n",
           std::min(static_cast<unsigned int>(inference_params.thread_pool_size),
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
  cache_config_.use_hctr_cache_implementation = inference_params.use_hctr_cache_implementation;
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
                 "\"%zu\" and %d will be used as the default embedding cache size.\n",
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
    lookup_handle_map_vec_.reserve(cache_config_.num_emb_table_);
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      // need to do init
      typename EmbedCache<TypeHashKey, TypeHashKey>::CacheConfig config;
      config.cacheSzInBytes = SLAB_SIZE * SET_ASSOCIATIVITY * cache_config_.num_set_in_cache_[i] *
                              cache_config_.embedding_vec_size_[i] * sizeof(float);
      HCTR_LOG(INFO, ROOT, "cache size in bytes: %zu\n", config.cacheSzInBytes);
      config.embedWidth = cache_config_.embedding_vec_size_[i] * sizeof(float);
      config.maxUpdateSampleSz =
          cache_config_.num_set_in_cache_[i] * EmbedCache<TypeHashKey, TypeHashKey>::NUM_WAYS;
      config.numTables = 1;
      gpu_emb_caches_.emplace_back(
          std::make_unique<EmbedCache<TypeHashKey, TypeHashKey>>(&allocator_, &logger_, config));
      gpu_emb_caches_.back()->Init();
      std::map<cudaStream_t, PerStreamLookupData> lookup_map;
      lookup_handle_map_vec_.emplace_back(lookup_map);
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

      // handle private data
      const int max_num_cache_set = cache_config_.num_set_in_cache_[i];
      const size_t max_num_keys = (SLAB_SIZE * SET_ASSOCIATIVITY) * max_num_cache_set;
      const size_t max_num_key_in_buffer =
          std::max(float(SLAB_SIZE * SET_ASSOCIATIVITY),
                   cache_config_.cache_refresh_percentage_per_iteration * max_num_keys);
      RefreshPrivateData ref_private_data;
      HCTR_LIB_THROW(
          cudaMallocHost(&ref_private_data.ptr_, max_num_key_in_buffer * sizeof(TypeHashKey)));
      HCTR_LIB_THROW(cudaEventCreate(&ref_private_data.modify_event_));
      gpu_emb_caches_[i]->ModifyContextCreate(ref_private_data.modify_handle_,
                                              max_num_key_in_buffer);
      refresh_private_data_.push_back(ref_private_data);
    }
  }
}
template <typename TypeHashKey>
typename EmbeddingCacheStoch<TypeHashKey>::PerStreamLookupData
EmbeddingCacheStoch<TypeHashKey>::GetLookupData(cudaStream_t stream, size_t table_id) {
  // need to readlock
  // it = lookup_map.find(stream)
  // release lock
  // if (it == end)
  //   write lock
  //   search again see if we already have
  //   if (it == end)
  //       create new context
  PerStreamLookupData ret;
  auto& lookup_handle_map = lookup_handle_map_vec_[table_id];
  bool needToInit = false;
  {
    ReadLock l(lookup_map_read_write_lock_);
    if (lookup_handle_map.find(stream) == lookup_handle_map.end()) {
      needToInit = true;
    } else {
      ret = lookup_handle_map[stream];
    }
  }

  if (needToInit) {
    WriteLock l(lookup_map_read_write_lock_);
    // search again if things were changed between locks
    if (lookup_handle_map.find(stream) == lookup_handle_map.end()) {
      PerStreamLookupData ret;
      HCTR_LIB_THROW(cudaEventCreate(&ret.event));
      lookup_handle_map[stream] = ret;
    }
    ret = lookup_handle_map[stream];
  }

  return ret;
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::SyncForModify(size_t table_id) const {
  // technically no need to lock here but since modify and query is mutually exclusive i don't know
  // who else is going to call me it better to be safe

  ReadLock l(lookup_map_read_write_lock_);
  for (auto& psl : lookup_handle_map_vec_[table_id]) {
    HCTR_LIB_THROW(cudaEventSynchronize(psl.second.event));
  }
}

template <typename TypeHashKey>
EmbeddingCacheStoch<TypeHashKey>::~EmbeddingCacheStoch() {
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

    for (auto& lookup_handle_map : lookup_handle_map_vec_) {
      for (auto& lookup : lookup_handle_map) {
        cudaEventDestroy(lookup.second.event);
      }
    }
    lookup_handle_map_vec_.clear();

    for (size_t i = 0; i < refresh_private_data_.size(); i++) {
      auto& r_private = refresh_private_data_[i];
      cudaFreeHost(r_private.ptr_);
      gpu_emb_caches_[i]->ModifyContextDestroy(r_private.modify_handle_);
      cudaEventSynchronize(r_private.modify_event_);
      cudaEventDestroy(r_private.modify_event_);
    }
    refresh_private_data_.clear();

    gpu_emb_caches_.clear();

    // clear vectors
  }
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::lookup(size_t const table_id, float* const d_vectors,
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
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::lookup_from_device(
    size_t const table_id, float* const d_vectors, const void* const d_keys, size_t const num_keys,
    float const hit_rate_threshold, cudaStream_t stream) {
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
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::Query(const TypeHashKey* d_keys, const size_t len,
                                             float* d_values, uint64_t* d_missing_index,
                                             TypeHashKey* d_missing_keys, size_t* d_missing_len,
                                             LookupContextHandle lookup_handle, cudaStream_t stream,
                                             size_t table_id) {
  ReadLock l(read_write_lock_);
  auto hLookup = GetLookupData(stream, table_id);
  // call cache Query
  gpu_emb_caches_[table_id]->Lookup(
      lookup_handle, d_keys, len, (int8_t*)d_values, d_missing_index, d_missing_keys, d_missing_len,
      0, cache_config_.embedding_vec_size_[table_id] * sizeof(float), stream);
  HCTR_LIB_THROW(cudaEventRecord(hLookup.event, stream));
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::lookup_from_device(
    size_t const table_id, float* const d_vectors, MemoryBlock* memory_block, size_t const num_keys,
    float const hit_rate_threshold, cudaStream_t stream) {
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    BaseUnit* start = profiler::start();
    // Unique
    const bool m_bUnique = false;
    if (m_bUnique) {
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
    }
    ec_profiler_->end(start, "Deduplicate the input embedding key for Embedding Cache");

    // Query
    const size_t query_length = m_bUnique ? workspace_handler.h_unique_length_[table_id] : num_keys;
    WorkspacePrivateData* private_data =
        reinterpret_cast<WorkspacePrivateData*>(workspace_handler.private_data_[table_id]);
    start = profiler::start();
    if (m_bUnique) {
      Query(
          static_cast<TypeHashKey*>(workspace_handler.d_unique_output_embeddingcolumns_[table_id]),
          workspace_handler.h_unique_length_[table_id], workspace_handler.d_hit_emb_vec_[table_id],
          workspace_handler.d_missing_index_[table_id],
          static_cast<TypeHashKey*>(workspace_handler.d_missing_embeddingcolumns_[table_id]),
          workspace_handler.d_missing_length_ + table_id, private_data->lookup_handle_, stream,
          table_id);
      HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.h_missing_length_ + table_id,
                                     workspace_handler.d_missing_length_ + table_id, sizeof(size_t),
                                     cudaMemcpyDeviceToHost, stream));
      // Set async flag
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    } else {
      Query(static_cast<TypeHashKey*>(workspace_handler.d_embeddingcolumns_[table_id]),
            query_length, d_vectors, workspace_handler.d_missing_index_[table_id],
            static_cast<TypeHashKey*>(workspace_handler.d_missing_embeddingcolumns_[table_id]),
            workspace_handler.d_missing_length_ + table_id, private_data->lookup_handle_, stream,
            table_id);
      HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.h_missing_length_ + table_id,
                                     workspace_handler.d_missing_length_ + table_id, sizeof(size_t),
                                     cudaMemcpyDeviceToHost, stream));
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
    ec_profiler_->end(start, "Native Embedding Cache Query API");
    if (query_length == 0) {
      workspace_handler.h_hit_rate_[table_id] = 1.0;
    } else {
      workspace_handler.h_hit_rate_[table_id] =
          1.0 - (static_cast<double>(workspace_handler.h_missing_length_[table_id]) /
                 (m_bUnique ? static_cast<double>(workspace_handler.h_unique_length_[table_id])
                            : query_length));
    }

    bool insert_flag{workspace_handler.h_hit_rate_[table_id] < hit_rate_threshold};
    start = profiler::start(workspace_handler.h_hit_rate_[table_id], ProfilerType_t::Occupancy);
    ec_profiler_->end(start, "The hit rate of Embedding Cache", ProfilerType_t::Occupancy);
    // Handle the missing keys

    start = profiler::start();
    // this will query the missing vectors from HPS, placed in workspace_handler->d_missing_vec
    // not this function will do a lot of copies, this all put the missing vectors in embedding
    // cache
    parameter_server_->insert_embedding_cache(table_id, this->shared_from_this(), workspace_handler,
                                              stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    if (insert_flag) {
      std::lock_guard<std::mutex> lock(mutex_);
      insert_workers_.submit([this, self(this->shared_from_this()), table_id, memory_block]() {
        parameter_server_insert_thread_func_2_<TypeHashKey>(table_id, parameter_server_, self,
                                                            memory_block, insert_streams_[table_id],
                                                            stream_mutex_);
      });
    }
    // Wait for memory copy to complete

    ec_profiler_->end(start, "Missing key synchronization insert into Embedding Cache");
    start = profiler::start();
    merge_emb_vec_async(d_vectors, workspace_handler.d_missing_emb_vec_[table_id],
                        workspace_handler.d_missing_index_[table_id],
                        workspace_handler.h_missing_length_[table_id],
                        cache_config_.embedding_vec_size_[table_id], BLOCK_SIZE_, stream);
    ec_profiler_->end(start, "Merge output from Embedding Cache", ProfilerType_t::Timeliness,
                      stream);
    start = profiler::start();
    if (m_bUnique) {
      // Decompress the hit emb_vec buffer to output buffer
      decompress_emb_vec_async(workspace_handler.d_hit_emb_vec_[table_id],
                               workspace_handler.d_unique_output_index_[table_id], d_vectors,
                               num_keys, cache_config_.embedding_vec_size_[table_id], BLOCK_SIZE_,
                               stream);
      // Clear the unique op object to be ready for next lookup
      static_cast<UniqueOp*>(workspace_handler.unique_op_obj_[table_id])->clear(stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }
    ec_profiler_->end(start, "decompress/deunique output from Embedding Cache");

    // Handle the missing keys, mode 2: synchronous
    if (!insert_flag) {
      parameter_server_->free_buffer(memory_block);
    }
  } else {
    HCTR_LOG_S(ERROR, WORLD)
        << "Cannot call internal lookup_from_device when disabling GPU embedding cache"
        << std::endl;
  }
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::Replace(const size_t table_id, const TypeHashKey* h_keys,
                                               const size_t len, const float* d_values,
                                               float hit_rate, ModifyContextHandle modify_handle,
                                               cudaEvent_t modify_event, cudaStream_t stream) {
  // Get a random number
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    gpu_emb_caches_[table_id]->ModifyContextSetReplaceData(
        modify_handle, h_keys, len, (const int8_t*)d_values, 0,
        cache_config_.embedding_vec_size_[table_id] * sizeof(float), false);
    {
      WriteLock l(read_write_lock_);
      SyncForModify(table_id);
      gpu_emb_caches_[table_id]->Modify(modify_handle, stream);
      HCTR_LIB_THROW(cudaEventRecord(modify_event, stream));
      HCTR_LIB_THROW(cudaEventSynchronize(modify_event));
    }
  }
}

// insert
template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::insert(const size_t table_id,
                                              EmbeddingCacheWorkspace& workspace_handler,
                                              cudaStream_t stream) {
  // don't do anything, this is called form HPS we will over ride this
  return;
}

// Todo: For iterative initialization of EC using the input space of the modelreader directly
template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::init(const size_t table_id,
                                            void* h_refresh_embeddingcolumns_,
                                            void* h_refresh_emb_vec_, float* h_quant_scales,
                                            size_t h_length_, cudaStream_t stream) {}

// at world start put something in cache
template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::init(const size_t table_id,
                                            EmbeddingCacheRefreshspace& refreshspace_handler,
                                            cudaStream_t stream) {
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);
    RefreshPrivateData private_data = refresh_private_data_[table_id];
    Replace(table_id, static_cast<TypeHashKey*>(refreshspace_handler.h_refresh_embeddingcolumns_),
            *refreshspace_handler.h_length_, refreshspace_handler.d_refresh_emb_vec_, 0.0,
            private_data.modify_handle_, private_data.modify_event_, stream);
  }
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::dump(const size_t table_id, void* const d_keys,
                                            size_t* const d_length, const size_t start_index,
                                            const size_t end_index, cudaStream_t stream) {
  // If GPU embedding cache is enabled
  HCTR_LOG_S(ERROR, WORLD) << "not imeplented" << std::endl;
  return;
}

// update function
template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::refresh(const size_t table_id, const void* const d_keys,
                                               const void* const d_vectors, const size_t length,
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
    WriteLock l(read_write_lock_);
    SyncForModify(table_id);
    HCTR_LIB_THROW(cudaMemcpyAsync(refresh_private_data_[table_id].ptr_, d_keys,
                                   length * sizeof(TypeHashKey), cudaMemcpyDefault, stream));
    cudaStreamSynchronize(stream);
    gpu_emb_caches_[table_id]->ModifyContextSetUpdateData(
        refresh_private_data_[table_id].modify_handle_, refresh_private_data_[table_id].ptr_,
        length, (const int8_t*)d_vectors, 0,
        cache_config_.embedding_vec_size_[table_id] * sizeof(float), false);

    gpu_emb_caches_[table_id]->Modify(refresh_private_data_[table_id].modify_handle_, stream);
    HCTR_LIB_THROW(cudaEventRecord(refresh_private_data_[table_id].modify_event_, stream));
    HCTR_LIB_THROW(cudaEventSynchronize(refresh_private_data_[table_id].modify_event_));

    ec_profiler_->end(start, "Refresh/Update exist embedding vector in Embedding cache",
                      ProfilerType_t::Timeliness, stream);
  }
}

// start thread waiting for update/insert
template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::finalize() {
  if (cache_config_.use_gpu_embedding_cache_) {
    std::lock_guard<std::mutex> lock(mutex_);

    CudaDeviceContext dev_restorer;
    dev_restorer.check_device(cache_config_.cuda_dev_id_);

    // Join insert threads
    insert_workers_.await_idle();
  }
}

template <typename TypeHashKey>
EmbeddingCacheWorkspace EmbeddingCacheStoch<TypeHashKey>::create_workspace() {
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
      // allocate cuBE workspace
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

    // allocate private data
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      WorkspacePrivateData* private_data_ptr = new WorkspacePrivateData;
      private_data_ptr->rd_.seed((uint64_t)private_data_ptr);

      gpu_emb_caches_[i]->ModifyContextCreate(private_data_ptr->modify_handle_,
                                              cache_config_.max_query_len_per_emb_table_[i]);
      gpu_emb_caches_[i]->LookupContextCreate(private_data_ptr->lookup_handle_, nullptr, 0);
      HCTR_LIB_THROW(cudaEventCreate(&private_data_ptr->modify_event_));
      workspace_handler.private_data_.push_back(private_data_ptr);
    }
  }
  return workspace_handler;
}

template <typename TypeHashKey>
void EmbeddingCacheStoch<TypeHashKey>::destroy_workspace(
    EmbeddingCacheWorkspace& workspace_handler) {
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

    // free private data
    for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
      WorkspacePrivateData* private_data_ptr =
          reinterpret_cast<WorkspacePrivateData*>(workspace_handler.private_data_[i]);

      gpu_emb_caches_[i]->ModifyContextDestroy(private_data_ptr->modify_handle_);
      gpu_emb_caches_[i]->LookupContextDestroy(private_data_ptr->lookup_handle_);
      HCTR_LIB_THROW(cudaEventSynchronize(private_data_ptr->modify_event_));
      HCTR_LIB_THROW(cudaEventDestroy(private_data_ptr->modify_event_));
      delete private_data_ptr;
    }
  }
}

template <typename TypeHashKey>
EmbeddingCacheRefreshspace EmbeddingCacheStoch<TypeHashKey>::create_refreshspace() {
  EmbeddingCacheRefreshspace refreshspace_handler;
  // If GPU embedding cache is enabled
  if (cache_config_.use_gpu_embedding_cache_) {
    // create cuBE refresh space
    const int max_num_cache_set = *max_element(cache_config_.num_set_in_cache_.begin(),
                                               cache_config_.num_set_in_cache_.end());
    const int max_embedding_size = *max_element(cache_config_.embedding_vec_size_.begin(),
                                                cache_config_.embedding_vec_size_.end());
    const size_t max_num_keys = (SLAB_SIZE * SET_ASSOCIATIVITY) * max_num_cache_set;
    const size_t max_num_key_in_buffer =
        std::max(float(SLAB_SIZE * SET_ASSOCIATIVITY),
                 cache_config_.cache_refresh_percentage_per_iteration * max_num_keys);
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
void EmbeddingCacheStoch<TypeHashKey>::destroy_refreshspace(
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

template class EmbeddingCacheStoch<long long>;
template class EmbeddingCacheStoch<unsigned int>;

}  // namespace HugeCTR
