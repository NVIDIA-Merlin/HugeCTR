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
#pragma once

#include <chrono>
#include <hps/lookup_session_base.hpp>
#include <thread_pool.hpp>

namespace HugeCTR {

class LookupSession : public LookupSessionBase {
 private:
  virtual void lookup_with_table_fusion_impl(const void* keys, float* d_vectors, size_t num_keys,
                                             size_t table_id, bool key_on_gpu,
                                             cudaStream_t stream) override final;
  virtual void lookup_from_device_impl(const void* d_keys, float* d_vectors, size_t num_keys,
                                       size_t table_id, cudaStream_t stream) override final;
  virtual void lookup_impl(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id, cudaStream_t stream) override final;

 public:
  virtual ~LookupSession();
  LookupSession(const InferenceParams& inference_params,
                const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
  LookupSession(LookupSession const&) = delete;
  LookupSession& operator=(LookupSession const&) = delete;

  virtual void lookup(const void* h_keys, float* d_vectors, size_t num_keys,
                      size_t table_id) override final;
  virtual void lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                      const size_t table_id, cudaStream_t stream) override final;
  virtual void lookup(const std::vector<const void*>& h_keys_per_table,
                      const std::vector<float*>& d_vectors_per_table,
                      const std::vector<size_t>& num_keys_per_table) override final;
  virtual void lookup_from_device(const void* d_keys, float* d_vectors, size_t num_keys,
                                  size_t table_id) override final;
  virtual void lookup_from_device(const void* d_keys, float* d_vectors, size_t num_keys,
                                  size_t table_id, cudaStream_t stream) override final;
  virtual void lookup_from_device(const std::vector<const void*>& d_keys_per_table,
                                  const std::vector<float*>& d_vectors_per_table,
                                  const std::vector<size_t>& num_keys_per_table) override final;

  virtual const InferenceParams get_inference_params() const override { return inference_params_; }
  virtual void set_profiler(int interation, int warmup, bool enable_bench) {
    ls_profiler_->set_config(interation, warmup, enable_bench);
  };
  virtual void profiler_print() { ls_profiler_->print(); };

 private:
  std::vector<cudaStream_t> lookup_streams_;
  std::shared_ptr<EmbeddingCacheBase> embedding_cache_;
  InferenceParams inference_params_;
  std::unique_ptr<profiler> ls_profiler_;
  std::mutex mutex_;
  std::condition_variable cv_;

  std::vector<void*> key_buffer_for_each_fused_table_;
  std::vector<float*> vec_buffer_for_each_fused_table_;
  std::vector<std::vector<size_t>> key_buffer_offset_for_each_fused_table_;
  std::vector<std::vector<size_t>> vec_buffer_offset_for_each_fused_table_;
  std::vector<std::vector<size_t>> num_keys_of_original_tables_for_each_fused_table_;

  std::vector<size_t> num_original_tables_in_each_fused_table_;
  std::vector<size_t> counter_for_each_fused_table_;
  std::vector<size_t> copy_key_counter_for_each_fused_table_;
  std::vector<size_t> copy_vec_counter_for_each_fused_table_;
  std::vector<int> ready_to_copy_key_for_each_fused_table_;
  std::vector<int> ready_to_copy_vec_for_each_fused_table_;

  ThreadPool table_fusion_thread_pool_;
  const std::chrono::milliseconds wait_duration_{1000};
};

}  // namespace HugeCTR