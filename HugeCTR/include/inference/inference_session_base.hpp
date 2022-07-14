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
#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <memory>

namespace HugeCTR {

class InferenceSessionBase {
 public:
  virtual ~InferenceSessionBase() = 0;
  InferenceSessionBase() = default;
  InferenceSessionBase(InferenceSessionBase const&) = delete;
  InferenceSessionBase& operator=(InferenceSessionBase const&) = delete;

  virtual void predict(float* d_dense, void* h_embeddingcolumns, int* d_row_ptrs, float* d_output,
                       int num_samples) = 0;

  static std::shared_ptr<InferenceSessionBase> create(
      const std::string& model_config_path, const InferenceParams& inference_params,
      const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
};

}  // namespace HugeCTR