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

#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <memory>

namespace HugeCTR {

class LookupSessionBase {
 private:
  virtual void lookup_with_table_fusion_impl(const void* keys, float* d_vectors, size_t num_keys,
                                             size_t table_id, bool key_on_gpu,
                                             cudaStream_t stream) = 0;
  virtual void lookup_from_device_impl(const void* d_keys, float* d_vectors, size_t num_keys,
                                       size_t table_id, cudaStream_t stream) = 0;
  virtual void lookup_impl(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id, cudaStream_t stream) = 0;

 public:
  virtual ~LookupSessionBase() = 0;
  LookupSessionBase() = default;
  LookupSessionBase(LookupSessionBase const&) = delete;
  LookupSessionBase& operator=(LookupSessionBase const&) = delete;

  virtual void lookup(const void* h_keys, float* d_vectors, size_t num_keys, size_t table_id) = 0;
  virtual void lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                      const size_t table_id, cudaStream_t stream) = 0;
  virtual void lookup(const std::vector<const void*>& h_keys_per_table,
                      const std::vector<float*>& d_vectors_per_table,
                      const std::vector<size_t>& num_keys_per_table) = 0;
  virtual void lookup_from_device(const void* d_keys, float* d_vectors, size_t num_keys,
                                  size_t table_id) = 0;
  virtual void lookup_from_device(const void* d_keys, float* d_vectors, size_t num_keys,
                                  size_t table_id, cudaStream_t stream) = 0;
  virtual void lookup_from_device(const std::vector<const void*>& d_keys_per_table,
                                  const std::vector<float*>& d_vectors_per_table,
                                  const std::vector<size_t>& num_keys_per_table) = 0;
  virtual const InferenceParams get_inference_params() const = 0;

  static std::shared_ptr<LookupSessionBase> create(
      const InferenceParams& inference_params,
      const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
  virtual void set_profiler(int interation, int warmup, bool enable_bench) = 0;
  virtual void profiler_print() = 0;
};

}  // namespace HugeCTR