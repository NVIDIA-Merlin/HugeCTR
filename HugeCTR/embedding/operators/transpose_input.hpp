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

#include <core/buffer.hpp>
#include <core/registry.hpp>
#include <embedding/common.hpp>

namespace embedding {

class PreprocessInput {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  EmbeddingLayout input_layout_;
  int num_lookup_;

  Tensor feature_major_key_;
  Tensor feature_major_bucket_range_;
  Tensor d_temp_scan_storage_;

 public:
  PreprocessInput(std::shared_ptr<CoreResourceManager> core,
                  const EmbeddingCollectionParam &ebc_param);

  void compute(const Tensor &key, const Tensor &bucket_range, Tensor *feature_major_key,
               Tensor *feature_major_bucket_range, int batch_size);
};

}  // namespace embedding