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

#include <hps/lookup_session.hpp>
#include <utils.hpp>

namespace HugeCTR {

LookupSessionBase::~LookupSessionBase() = default;

std::shared_ptr<LookupSessionBase> LookupSessionBase::create(
    const InferenceParams& inference_params,
    const std::shared_ptr<EmbeddingCacheBase>& embedding_cache) {
  return std::make_shared<LookupSession>(inference_params, embedding_cache);
}

LookupSession::LookupSession(const InferenceParams& inference_params,
                             const std::shared_ptr<EmbeddingCacheBase>& embedding_cache)
    : LookupSessionBase(),
      embedding_cache_(embedding_cache),
      inference_params_(inference_params),
      table_fusion_thread_pool_("table fusion",
                                inference_params.original_table_id_to_fused_table_id_map.size()) {
  try {
    auto b2s = [](const char val) { return val ? "True" : "False"; };
    HCTR_LOG_S(INFO, WORLD) << "LookupSession i64_input_key: "
                            << b2s(inference_params_.i64_input_key) << std::endl;
    if (inference_params_.use_gpu_embedding_cache &&
        embedding_cache_->get_device_id() != inference_params_.device_id) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "The device id of inference_params is not consistent with that of embedding cache.");
    }
    HCTR_LOG(INFO, ROOT, "Creating lookup session for %s on device: %d\n",
             inference_params_.model_name.c_str(), inference_params_.device_id);
    ls_profiler_ = std::make_unique<profiler>(ProfilerTarget_t::LOOKSESSION);
    size_t num_tables = inference_params_.fuse_embedding_table
                            ? inference_params_.fused_sparse_model_files.size()
                            : inference_params_.sparse_model_files.size();

    if (inference_params_.fuse_embedding_table) {
      num_keys_of_original_tables_for_each_fused_table_.resize(num_tables);
      key_buffer_offset_for_each_fused_table_.resize(num_tables);
      vec_buffer_offset_for_each_fused_table_.resize(num_tables);
      for (size_t fused_id{0}; fused_id < num_tables; ++fused_id) {
        size_t current_num_tables =
            inference_params_.fused_table_id_to_original_table_id_map[fused_id].size();
        num_original_tables_in_each_fused_table_.push_back(current_num_tables);
        num_keys_of_original_tables_for_each_fused_table_[fused_id].resize(current_num_tables);
        key_buffer_offset_for_each_fused_table_[fused_id].resize(current_num_tables + 1);
        vec_buffer_offset_for_each_fused_table_[fused_id].resize(current_num_tables + 1);
        ready_to_copy_key_for_each_fused_table_.push_back(0);
        ready_to_copy_vec_for_each_fused_table_.push_back(0);
      }
    }
    counter_for_each_fused_table_ = num_original_tables_in_each_fused_table_;
    copy_key_counter_for_each_fused_table_ = num_original_tables_in_each_fused_table_;
    copy_vec_counter_for_each_fused_table_ = num_original_tables_in_each_fused_table_;

    CudaDeviceContext dev_restorer;
    dev_restorer.set_device(inference_params_.device_id);
    for (size_t idx = 0; idx < num_tables; ++idx) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      lookup_streams_.push_back(stream);
    }
    if (inference_params_.fuse_embedding_table) {
      size_t key_size_in_byte =
          inference_params_.i64_input_key ? sizeof(long long) : sizeof(unsigned int);
      for (size_t fused_id{0}; fused_id < num_tables; ++fused_id) {
        void* current_key_buffer;
        float* current_vec_buffer;
        size_t max_num_key_per_sample =
            inference_params_.maxnum_catfeature_query_per_table_per_sample[fused_id];
        size_t emb_vec_size = inference_params_.embedding_vecsize_per_table[fused_id];
        HCTR_LIB_THROW(cudaMalloc(
            &current_key_buffer,
            inference_params_.max_batchsize * max_num_key_per_sample * key_size_in_byte));
        HCTR_LIB_THROW(cudaMalloc(reinterpret_cast<void**>(&current_vec_buffer),
                                  inference_params_.max_batchsize * max_num_key_per_sample *
                                      emb_vec_size * sizeof(float)));
        key_buffer_for_each_fused_table_.push_back(current_key_buffer);
        vec_buffer_for_each_fused_table_.push_back(current_vec_buffer);
      }
    }

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  return;
}

LookupSession::~LookupSession() {
  CudaDeviceContext dev_restorer;
  dev_restorer.set_device(inference_params_.device_id);
  for (auto stream : lookup_streams_) HCTR_LIB_THROW(cudaStreamDestroy(stream));
  if (inference_params_.fuse_embedding_table) {
    size_t num_tables = inference_params_.fused_sparse_model_files.size();
    for (size_t fused_id{0}; fused_id < num_tables; ++fused_id) {
      HCTR_LIB_THROW(cudaFree(key_buffer_for_each_fused_table_[fused_id]));
      key_buffer_for_each_fused_table_[fused_id] = nullptr;
      HCTR_LIB_THROW(cudaFree(vec_buffer_for_each_fused_table_[fused_id]));
      vec_buffer_for_each_fused_table_[fused_id] = nullptr;
    }
  }
}

void LookupSession::lookup_with_table_fusion_impl(const void* keys, float* d_vectors,
                                                  size_t num_keys, size_t table_id, bool key_on_gpu,
                                                  cudaStream_t stream) {
  size_t fused_table_id = inference_params_.original_table_id_to_fused_table_id_map[table_id];
  auto original_table_id_list =
      inference_params_.fused_table_id_to_original_table_id_map[fused_table_id];
  size_t idx_in_list =
      find(original_table_id_list.begin(), original_table_id_list.end(), table_id) -
      original_table_id_list.begin();
  size_t key_size_in_byte =
      inference_params_.i64_input_key ? sizeof(long long) : sizeof(unsigned int);

  // Decrement the counter for the current fused table
  {
    std::unique_lock lock(mutex_);
    counter_for_each_fused_table_[fused_table_id] -= 1;
    num_keys_of_original_tables_for_each_fused_table_[fused_table_id][idx_in_list] = num_keys;
    cv_.notify_all();
  }

  // Wait for inputs of all tables for the current fused table
  {
    if (table_id ==
        inference_params_.fused_table_id_to_original_table_id_map[fused_table_id].back()) {
      std::unique_lock lock(mutex_);
      if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
            return counter_for_each_fused_table_[fused_table_id] == 0;
          })) {
        size_t current_num_tables = num_original_tables_in_each_fused_table_[fused_table_id];
        key_buffer_offset_for_each_fused_table_[fused_table_id][0] = 0;
        vec_buffer_offset_for_each_fused_table_[fused_table_id][0] = 0;
        for (size_t idx{0}; idx < current_num_tables; ++idx) {
          key_buffer_offset_for_each_fused_table_[fused_table_id][idx + 1] =
              key_buffer_offset_for_each_fused_table_[fused_table_id][idx] +
              num_keys_of_original_tables_for_each_fused_table_[fused_table_id][idx] *
                  key_size_in_byte;
          vec_buffer_offset_for_each_fused_table_[fused_table_id][idx + 1] =
              vec_buffer_offset_for_each_fused_table_[fused_table_id][idx] +
              num_keys_of_original_tables_for_each_fused_table_[fused_table_id][idx] *
                  inference_params_.embedding_vecsize_per_table[fused_table_id] * sizeof(float);
        }
        ready_to_copy_key_for_each_fused_table_[fused_table_id] = 1;
        cv_.notify_all();
      } else {
        HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                    "multithreading for embedding lookup."
                                 << std::endl;
        return;
      }
    }
  }

  // Copy from keys to key buffer of current fused table
  {
    std::unique_lock lock(mutex_);
    if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
          return ready_to_copy_key_for_each_fused_table_[fused_table_id] == 1;
        })) {
      if (key_on_gpu) {
        HCTR_LIB_THROW(cudaMemcpyAsync(
            reinterpret_cast<char*>(key_buffer_for_each_fused_table_[fused_table_id]) +
                key_buffer_offset_for_each_fused_table_[fused_table_id][idx_in_list],
            keys, num_keys * key_size_in_byte, cudaMemcpyDeviceToDevice, stream));
      } else {
        HCTR_LIB_THROW(cudaMemcpyAsync(
            reinterpret_cast<char*>(key_buffer_for_each_fused_table_[fused_table_id]) +
                key_buffer_offset_for_each_fused_table_[fused_table_id][idx_in_list],
            keys, num_keys * key_size_in_byte, cudaMemcpyHostToDevice, stream));
      }
      copy_key_counter_for_each_fused_table_[fused_table_id] -= 1;
      cv_.notify_all();
    } else {
      HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                  "multithreading for embedding lookup."
                               << std::endl;
      return;
    }
  }

  // Perform embedding lookup for current fused table
  {
    if (table_id ==
        inference_params_.fused_table_id_to_original_table_id_map[fused_table_id].back()) {
      std::unique_lock lock(mutex_);
      if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
            return copy_key_counter_for_each_fused_table_[fused_table_id] == 0;
          })) {
        size_t fused_num_keys =
            key_buffer_offset_for_each_fused_table_[fused_table_id][idx_in_list + 1] /
            key_size_in_byte;
        this->lookup_from_device_impl(key_buffer_for_each_fused_table_[fused_table_id],
                                      vec_buffer_for_each_fused_table_[fused_table_id],
                                      fused_num_keys, fused_table_id, stream);
        ready_to_copy_vec_for_each_fused_table_[fused_table_id] = 1;
        cv_.notify_all();
      } else {
        HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                    "multithreading for embedding lookup."
                                 << std::endl;
        return;
      }
    }
  }

  // Copy from vector buffer of current fused table to d_vectors
  {
    std::unique_lock lock(mutex_);
    if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
          return ready_to_copy_vec_for_each_fused_table_[fused_table_id] == 1;
        })) {
      HCTR_LIB_THROW(cudaMemcpyAsync(
          d_vectors,
          reinterpret_cast<char*>(vec_buffer_for_each_fused_table_[fused_table_id]) +
              vec_buffer_offset_for_each_fused_table_[fused_table_id][idx_in_list],
          num_keys * inference_params_.embedding_vecsize_per_table[fused_table_id] * sizeof(float),
          cudaMemcpyDeviceToDevice, stream));
      copy_vec_counter_for_each_fused_table_[fused_table_id] -= 1;
      cv_.notify_all();
    } else {
      HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                  "multithreading for embedding lookup."
                               << std::endl;
      return;
    }
  }

  // Reset counters and flags
  {
    if (table_id ==
        inference_params_.fused_table_id_to_original_table_id_map[fused_table_id].back()) {
      std::unique_lock lock(mutex_);
      if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
            return copy_vec_counter_for_each_fused_table_[fused_table_id] == 0;
          })) {
        counter_for_each_fused_table_[fused_table_id] =
            num_original_tables_in_each_fused_table_[fused_table_id];
        copy_key_counter_for_each_fused_table_[fused_table_id] =
            num_original_tables_in_each_fused_table_[fused_table_id];
        copy_vec_counter_for_each_fused_table_[fused_table_id] =
            num_original_tables_in_each_fused_table_[fused_table_id];
        ready_to_copy_key_for_each_fused_table_[fused_table_id] = 0;
        ready_to_copy_vec_for_each_fused_table_[fused_table_id] = 0;
        cv_.notify_all();
      } else {
        HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                    "multithreading for embedding lookup."
                                 << std::endl;
        return;
      }
    }
  }

  {
    std::unique_lock lock(mutex_);
    if (cv_.wait_for(lock, wait_duration_, [this, fused_table_id] {
          return counter_for_each_fused_table_[fused_table_id] ==
                 num_original_tables_in_each_fused_table_[fused_table_id];
        })) {
      HCTR_LOG_S(TRACE, WORLD) << "Finish embedding lookup for original table id: " << table_id
                               << std::endl;
    } else {
      HCTR_LOG_S(ERROR, WORLD) << "Time out. The fusing table feature of HPS requires CPU "
                                  "multithreading for embedding lookup."
                               << std::endl;
      return;
    }
  }
}

void LookupSession::lookup_from_device_impl(const void* d_keys, float* d_vectors, size_t num_keys,
                                            size_t table_id, cudaStream_t stream) {
  CudaDeviceContext dev_restorer;
  dev_restorer.set_device(inference_params_.device_id);
  embedding_cache_->lookup_from_device(table_id, d_vectors, d_keys, num_keys,
                                       inference_params_.hit_rate_threshold, stream);
}

void LookupSession::lookup_impl(const void* const h_keys, float* const d_vectors,
                                const size_t num_keys, const size_t table_id, cudaStream_t stream) {
  CudaDeviceContext dev_restorer;
  dev_restorer.set_device(inference_params_.device_id);
  embedding_cache_->lookup(table_id, d_vectors, h_keys, num_keys,
                           inference_params_.hit_rate_threshold, stream);
}

void LookupSession::lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id, cudaStream_t stream) {
  if (inference_params_.fuse_embedding_table) {
    this->lookup_with_table_fusion_impl(h_keys, d_vectors, num_keys, table_id, false, stream);
  } else {
    this->lookup_impl(h_keys, d_vectors, num_keys, table_id, stream);
  }
}

void LookupSession::lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id) {
  const auto begin = std::chrono::high_resolution_clock::now();
  BaseUnit* start = profiler::start();

  if (inference_params_.fuse_embedding_table) {
    size_t fused_table_id = inference_params_.original_table_id_to_fused_table_id_map[table_id];
    this->lookup_with_table_fusion_impl(h_keys, d_vectors, num_keys, table_id, false,
                                        lookup_streams_[fused_table_id]);
    HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[fused_table_id]));
  } else {
    this->lookup_impl(h_keys, d_vectors, num_keys, table_id, lookup_streams_[table_id]);
    HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[table_id]));
  }

  ls_profiler_->end(start, "End-to-end lookup embedding keys for Lookup session");
  const auto latency = std::chrono::high_resolution_clock::now() - begin;
  HCTR_LOG_S(TRACE, WORLD) << "Lookup single table; number of keys " << num_keys << ", table id  "
                           << table_id << "lookup latency: " << latency.count() / 1000 << " us."
                           << std::endl;
}

void LookupSession::lookup(const std::vector<const void*>& h_keys_per_table,
                           const std::vector<float*>& d_vectors_per_table,
                           const std::vector<size_t>& num_keys_per_table) {
  size_t original_num_tables =
      inference_params_.fuse_embedding_table
          ? inference_params_.original_table_id_to_fused_table_id_map.size()
          : inference_params_.sparse_model_files.size();
  HCTR_CHECK_HINT(h_keys_per_table.size() == original_num_tables,
                  "The h_keys_per_table.size() should be equal to the number of embedding tables");
  HCTR_CHECK_HINT(
      d_vectors_per_table.size() == original_num_tables,
      "The d_vectors_per_table.size() should be equal to the number of embedding tables");

  const auto begin = std::chrono::high_resolution_clock::now();
  BaseUnit* start = profiler::start();

  for (size_t table_id{0}; table_id < original_num_tables; ++table_id) {
    if (inference_params_.fuse_embedding_table) {
      auto work_func = [this, h_keys_per_table, d_vectors_per_table, num_keys_per_table,
                        table_id]() {
        size_t fused_table_id = inference_params_.original_table_id_to_fused_table_id_map[table_id];
        this->lookup_with_table_fusion_impl(
            h_keys_per_table[table_id], d_vectors_per_table[table_id], num_keys_per_table[table_id],
            table_id, false, lookup_streams_[fused_table_id]);
      };
      table_fusion_thread_pool_.submit(work_func);
    } else {
      this->lookup_impl(h_keys_per_table[table_id], d_vectors_per_table[table_id],
                        num_keys_per_table[table_id], table_id, lookup_streams_[table_id]);
    }
  }
  if (inference_params_.fuse_embedding_table) {
    table_fusion_thread_pool_.await_idle();
  }
  for (auto stream : lookup_streams_) {
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }

  ls_profiler_->end(start, "End-to-end lookup embedding keys from multi-table Lookup session");
  const auto latency = std::chrono::high_resolution_clock::now() - begin;
  HCTR_LOG_S(TRACE, WORLD) << "Lookup multiple tables;"
                           << "lookup latency: " << latency.count() / 1000 << " us." << std::endl;
}

void LookupSession::lookup_from_device(const void* d_keys, float* d_vectors, size_t num_keys,
                                       size_t table_id, cudaStream_t stream) {
  if (inference_params_.fuse_embedding_table) {
    this->lookup_with_table_fusion_impl(d_keys, d_vectors, num_keys, table_id, true, stream);
  } else {
    this->lookup_from_device_impl(d_keys, d_vectors, num_keys, table_id, stream);
  }
}

void LookupSession::lookup_from_device(const void* const d_keys, float* const d_vectors,
                                       const size_t num_keys, const size_t table_id) {
  const auto begin = std::chrono::high_resolution_clock::now();
  BaseUnit* start = profiler::start();

  if (inference_params_.fuse_embedding_table) {
    size_t fused_table_id = inference_params_.original_table_id_to_fused_table_id_map[table_id];
    this->lookup_with_table_fusion_impl(d_keys, d_vectors, num_keys, table_id, true,
                                        lookup_streams_[fused_table_id]);
    HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[fused_table_id]));
  } else {
    this->lookup_from_device_impl(d_keys, d_vectors, num_keys, table_id, lookup_streams_[table_id]);
    HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[table_id]));
  }

  ls_profiler_->end(start, "End-to-end lookup embedding keys for Lookup session");
  const auto latency = std::chrono::high_resolution_clock::now() - begin;
  HCTR_LOG_S(TRACE, WORLD) << "Lookup single table; number of keys " << num_keys << ", table id  "
                           << table_id << "lookup latency: " << latency.count() / 1000 << " us."
                           << std::endl;
}

void LookupSession::lookup_from_device(const std::vector<const void*>& d_keys_per_table,
                                       const std::vector<float*>& d_vectors_per_table,
                                       const std::vector<size_t>& num_keys_per_table) {
  size_t original_num_tables =
      inference_params_.fuse_embedding_table
          ? inference_params_.original_table_id_to_fused_table_id_map.size()
          : inference_params_.sparse_model_files.size();
  HCTR_CHECK_HINT(d_keys_per_table.size() == original_num_tables,
                  "The d_keys_per_table.size() should be equal to the number of embedding tables");
  HCTR_CHECK_HINT(
      d_vectors_per_table.size() == original_num_tables,
      "The d_vectors_per_table.size() should be equal to the number of embedding tables");

  const auto begin = std::chrono::high_resolution_clock::now();
  BaseUnit* start = profiler::start();

  for (size_t table_id{0}; table_id < original_num_tables; ++table_id) {
    if (inference_params_.fuse_embedding_table) {
      auto work_func = [this, d_keys_per_table, d_vectors_per_table, num_keys_per_table,
                        table_id]() {
        size_t fused_table_id = inference_params_.original_table_id_to_fused_table_id_map[table_id];
        this->lookup_with_table_fusion_impl(
            d_keys_per_table[table_id], d_vectors_per_table[table_id], num_keys_per_table[table_id],
            table_id, true, lookup_streams_[fused_table_id]);
      };
      table_fusion_thread_pool_.submit(work_func);
    } else {
      this->lookup_from_device_impl(d_keys_per_table[table_id], d_vectors_per_table[table_id],
                                    num_keys_per_table[table_id], table_id,
                                    lookup_streams_[table_id]);
    }
  }
  if (inference_params_.fuse_embedding_table) {
    table_fusion_thread_pool_.await_idle();
  }
  for (auto stream : lookup_streams_) {
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }

  ls_profiler_->end(start, "End-to-end lookup embedding keys for multi-table Lookup session");
  const auto latency = std::chrono::high_resolution_clock::now() - begin;
  HCTR_LOG_S(TRACE, WORLD) << "Lookup multiple tables;"
                           << "lookup latency: " << latency.count() / 1000 << " us." << std::endl;
}

}  // namespace HugeCTR