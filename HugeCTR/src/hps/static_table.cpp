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
StaticTable<TypeHashKey>::StaticTable(const InferenceParams& inference_params,
                                      const parameter_server_config& ps_config,
                                      HierParameterServerBase* const parameter_server)
    : EmbeddingCacheBase(), parameter_server_(parameter_server) {
  // Store the configuration
  cache_config_.num_emb_table_ = inference_params.fuse_embedding_table
                                     ? inference_params.fused_sparse_model_files.size()
                                     : inference_params.sparse_model_files.size();
  cache_config_.cache_size_percentage_ = inference_params.cache_size_percentage;
  cache_config_.cache_refresh_percentage_per_iteration =
      inference_params.cache_refresh_percentage_per_iteration;
  cache_config_.default_value_for_each_table = inference_params.default_value_for_each_table;
  cache_config_.model_name_ = inference_params.model_name;
  cache_config_.cuda_dev_id_ = inference_params.device_id;
  cache_config_.use_gpu_embedding_cache_ = inference_params.use_gpu_embedding_cache;

  auto b2s = [](const char val) { return val ? "True" : "False"; };
  HCTR_LOG(INFO, ROOT, "Model name: %s\n", inference_params.model_name.c_str());
  HCTR_LOG(INFO, ROOT, "Max batch size: %lu\n", inference_params.max_batchsize);
  HCTR_LOG(INFO, ROOT, "Fuse embedding tables: %s\n", b2s(inference_params.fuse_embedding_table));
  HCTR_LOG(INFO, ROOT, "Number of embedding tables: %zu\n", cache_config_.num_emb_table_);
  HCTR_LOG(INFO, ROOT, "Use static table: %s\n", b2s(inference_params.use_static_table));
  HCTR_LOG(INFO, ROOT, "Use I64 input key: %s\n", b2s(inference_params.i64_input_key));
  HCTR_LOG(INFO, ROOT, "The size of worker memory pool: %u\n",
           inference_params.number_of_worker_buffers_in_pool);
  HCTR_LOG(INFO, ROOT, "The size of refresh memory pool: %u\n",
           inference_params.number_of_refresh_buffers_in_pool);
  HCTR_LOG(INFO, ROOT, "The refresh percentage : %f\n",
           inference_params.cache_refresh_percentage_per_iteration);

  // initialize the profiler
  ec_profiler_ = std::make_unique<profiler>(ProfilerTarget_t::EC);

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

  // This is the only two places to set the cuda context in static table
  CudaDeviceContext dev_restorer;
  dev_restorer.set_device(cache_config_.cuda_dev_id_);

  // Allocate resources.
  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    const size_t num_row = ps_config.embedding_key_count_.at(inference_params.model_name)[i];
    static_tables_.emplace_back(
        std::make_unique<Cache>(num_row, cache_config_.embedding_vec_size_[i],
                                cache_config_.default_value_for_each_table[i]));
    cache_config_.num_set_in_cache_.push_back(num_row);
  }

  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    refresh_streams_.push_back(stream);
  }
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::lookup(size_t table_id, float* d_vectors, const void* h_keys,
                                      size_t num_keys, float hit_rate_threshold,
                                      cudaStream_t stream) {
  MemoryBlock* memory_block = nullptr;
  BaseUnit* start = profiler::start();
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(parameter_server_->apply_buffer(
        cache_config_.model_name_, cache_config_.cuda_dev_id_, CACHE_SPACE_TYPE::WORKER));
  }
  ec_profiler_->end(start,
                    "Apply for workspace from the memory pool for Static Embedding Cache Lookup");
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);
  // Copy the keys to device
  start = profiler::start();
  HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_embeddingcolumns_[table_id], h_keys,
                                 num_keys * sizeof(TypeHashKey), cudaMemcpyHostToDevice, stream));
  ec_profiler_->end(start, "Copy the input to workspace of Static Embedding Cache",
                    ProfilerType_t::Timeliness, stream);
  start = profiler::start();
  lookup_from_device(table_id, d_vectors, memory_block, num_keys, stream);
  parameter_server_->free_buffer(memory_block);
  ec_profiler_->end(start, "Lookup the embedding keys from Static Embedding Cache",
                    ProfilerType_t::Timeliness, stream);
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::lookup_from_device(size_t table_id, float* d_vectors,
                                                  const void* d_keys, size_t num_keys,
                                                  float hit_rate_threshold, cudaStream_t stream) {
  MemoryBlock* memory_block = nullptr;
  BaseUnit* start = profiler::start();
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<struct MemoryBlock*>(parameter_server_->apply_buffer(
        cache_config_.model_name_, cache_config_.cuda_dev_id_, CACHE_SPACE_TYPE::WORKER));
  }
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;
  ec_profiler_->end(start,
                    "Apply for workspace from the memory pool for Static Embedding Cache Lookup");

  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);
  start = profiler::start();
  HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_embeddingcolumns_[table_id], d_keys,
                                 num_keys * sizeof(TypeHashKey), cudaMemcpyDeviceToDevice, stream));
  ec_profiler_->end(start, "Copy the input to workspace of Static Embedding Cache",
                    ProfilerType_t::Timeliness, stream);
  start = profiler::start();
  lookup_from_device(table_id, d_vectors, memory_block, num_keys, stream);
  ec_profiler_->end(start, "Lookup the embedding keys from Static Embedding Cache");
  parameter_server_->free_buffer(memory_block);
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::lookup_from_device(size_t table_id, float* d_vectors,
                                                  MemoryBlock* memory_block, size_t num_keys,
                                                  cudaStream_t stream) {
  EmbeddingCacheWorkspace workspace_handler = memory_block->worker_buffer;

  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);

  static_tables_[table_id]->Query(
      static_cast<TypeHashKey*>(workspace_handler.d_embeddingcolumns_[table_id]), num_keys,
      d_vectors, stream);
}

template <typename TypeHashKey>
EmbeddingCacheWorkspace StaticTable<TypeHashKey>::create_workspace() {
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
  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);

  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    void* d_embeddingcolumns;
    HCTR_LIB_THROW(cudaMalloc(&d_embeddingcolumns,
                              cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey)));
    workspace_handler.d_embeddingcolumns_.push_back(d_embeddingcolumns);

    uint64_t* d_unique_output_index;
    HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_unique_output_index),
                              cache_config_.max_query_len_per_emb_table_[i] * sizeof(uint64_t)));
    workspace_handler.d_unique_output_index_.push_back(d_unique_output_index);

    void* d_unique_output_embeddingcolumns;
    HCTR_LIB_THROW(cudaMalloc(&d_unique_output_embeddingcolumns,
                              cache_config_.max_query_len_per_emb_table_[i] * sizeof(TypeHashKey)));
    workspace_handler.d_unique_output_embeddingcolumns_.push_back(d_unique_output_embeddingcolumns);

    float* d_hit_emb_vec;
    HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&d_hit_emb_vec),
                              cache_config_.max_query_len_per_emb_table_[i] *
                                  cache_config_.embedding_vec_size_[i] * sizeof(float)));
    workspace_handler.d_hit_emb_vec_.push_back(d_hit_emb_vec);

    void* d_missing_embeddingcolumns;
    HCTR_LIB_THROW(cudaMalloc(&d_missing_embeddingcolumns,
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
  return workspace_handler;
}

template <typename TypeHashKey>
EmbeddingCacheRefreshspace StaticTable<TypeHashKey>::create_refreshspace() {
  EmbeddingCacheRefreshspace refreshspace_handler;
  const size_t max_num_keys =
      *max_element(cache_config_.num_set_in_cache_.begin(), cache_config_.num_set_in_cache_.end());
  const int max_embedding_size = *max_element(cache_config_.embedding_vec_size_.begin(),
                                              cache_config_.embedding_vec_size_.end());
  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);

  // Create memory buffers.
  HCTR_LIB_THROW(cudaHostAlloc(&refreshspace_handler.h_refresh_embeddingcolumns_,
                               max_num_keys * sizeof(TypeHashKey), cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&refreshspace_handler.h_refresh_emb_vec_),
                               max_num_keys * max_embedding_size * sizeof(float),
                               cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaHostAlloc(reinterpret_cast<void**>(&refreshspace_handler.h_length_),
                               sizeof(size_t), cudaHostAllocPortable));
  HCTR_LIB_THROW(cudaMalloc(&refreshspace_handler.d_refresh_embeddingcolumns_,
                            max_num_keys * sizeof(TypeHashKey)));
  HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&refreshspace_handler.d_refresh_emb_vec_),
                            max_num_keys * max_embedding_size * sizeof(float)));
  HCTR_LIB_THROW(
      cudaMalloc(reinterpret_cast<void**>(&refreshspace_handler.d_length_), sizeof(size_t)));
  return refreshspace_handler;
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::init(const size_t table_id,
                                    EmbeddingCacheRefreshspace& refreshspace_handler,
                                    cudaStream_t stream) {
  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);
  static_tables_[table_id]->Init(
      static_cast<TypeHashKey*>(refreshspace_handler.d_refresh_embeddingcolumns_),
      *refreshspace_handler.h_length_, refreshspace_handler.d_refresh_emb_vec_, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::refresh(const size_t table_id, const void* const d_keys,
                                       const float* const d_vectors, const size_t length,
                                       cudaStream_t stream) {
  CudaDeviceContext dev_restorer;
  dev_restorer.check_device(cache_config_.cuda_dev_id_);
  static_tables_[table_id]->Clear(stream);
  static_tables_[table_id]->Init(static_cast<const TypeHashKey*>(d_keys), length, d_vectors,
                                 stream);
}

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::destroy_workspace(EmbeddingCacheWorkspace& workspace_handler) {
  // Free common buffer.
  for (size_t i = 0; i < cache_config_.num_emb_table_; i++) {
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_embeddingcolumns_[i]));
    workspace_handler.h_embeddingcolumns_[i] = nullptr;
    HCTR_LIB_THROW(cudaFreeHost(workspace_handler.h_missing_emb_vec_[i]));
    workspace_handler.h_missing_emb_vec_[i] = nullptr;
  }
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

template <typename TypeHashKey>
void StaticTable<TypeHashKey>::destroy_refreshspace(
    EmbeddingCacheRefreshspace& refreshspace_handler) {
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

template <typename TypeHashKey>
StaticTable<TypeHashKey>::~StaticTable() {
  // This is the only two places to set the cuda context in static table
  CudaDeviceContext dev_restorer;
  dev_restorer.set_device(cache_config_.cuda_dev_id_);
  // Destroy resources.
  for (auto& stream : refresh_streams_) {
    cudaStreamDestroy(stream);
  }
  refresh_streams_.clear();
}

template class StaticTable<long long>;
template class StaticTable<unsigned int>;

}  // namespace HugeCTR