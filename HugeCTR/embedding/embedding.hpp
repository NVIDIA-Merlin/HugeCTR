/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <vector>

#include "embedding_table.hpp"

namespace embedding {

class IGroupedEmbeddingOp {
 public:
  virtual ~IGroupedEmbeddingOp() = default;

  virtual void forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                               ILookup *embedding_table, Tensor &output_buffer, int batch_size) = 0;

  virtual void backward_per_gpu(const Tensor &top_grad, bool do_allreduce, Tensor *unique_key,
                                size_t *num_unique_key, Tensor *num_unique_key_per_table_offset,
                                size_t *num_table_offset, Tensor *table_id_list, Tensor *wgrad,
                                Tensor *wgrad_idx_offset) = 0;
};

std::vector<std::unique_ptr<IGroupedEmbeddingOp>> create_grouped_embeddings(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);
}  // namespace embedding
