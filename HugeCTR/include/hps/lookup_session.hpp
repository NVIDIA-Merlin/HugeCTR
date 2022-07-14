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
#include <hps/lookup_session_base.hpp>

namespace HugeCTR {

class LookupSession : public LookupSessionBase {
 public:
  virtual ~LookupSession();
  LookupSession(const InferenceParams& inference_params,
                const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
  LookupSession(LookupSession const&) = delete;
  LookupSession& operator=(LookupSession const&) = delete;

  virtual void lookup(const void* h_keys, float* d_vectors, size_t num_keys, size_t table_id);
  virtual void lookup(const std::vector<const void*>& h_keys_per_table,
                      const std::vector<float*>& d_vectors_per_table,
                      const std::vector<size_t>& num_keys_per_table);

 private:
  std::vector<cudaStream_t> lookup_streams_;
  std::shared_ptr<EmbeddingCacheBase> embedding_cache_;
  InferenceParams inference_params_;
};

}  // namespace HugeCTR