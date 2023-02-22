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

#include <HugeCTR/embedding/common.hpp>
#include <core/buffer.hpp>
#include <core23/registry.hpp>

namespace embedding {
using core::CoreResourceManager;

class CompressOffset {
  std::shared_ptr<CoreResourceManager> core_;
  int num_compressed_offset_;
  core23::Tensor compressed_offset_;

 public:
  CompressOffset() = default;

  CompressOffset(std::shared_ptr<CoreResourceManager> core, int num_compressed_offset);

  void compute(const core23::Tensor &offset, int batch_size, core23::Tensor *compressed_offset);
};

class AverageCombiner {
  std::shared_ptr<CoreResourceManager> core_;
  int num_gpus_;
  int num_local_embedding_;

 public:
  core23::Tensor float_emb_vec_;

  AverageCombiner() = default;

  AverageCombiner(std::shared_ptr<CoreResourceManager> core, int num_gpus, int num_local_embedding,
                  const std::vector<int> &ev_size_list, int universal_batch_size);

  void compute_feature_major(const core23::Tensor &bucket_range, const core23::Tensor &src_emb_vec,
                             const core23::Tensor &d_local_embedding_list,
                             const core23::Tensor &d_combiner_list,
                             const core23::Tensor &d_ev_size_offset, int batch_size,
                             int max_ev_size);

  void compute_batch_major(const core23::Tensor &bucket_range, const core23::Tensor &src_emb_vec,
                           const core23::Tensor &d_local_embedding_list,
                           const core23::Tensor &d_combiner_list,
                           const core23::Tensor &d_ev_size_offset, int batch_size, int max_ev_size,
                           int num_lookup);
};

}  // namespace embedding
