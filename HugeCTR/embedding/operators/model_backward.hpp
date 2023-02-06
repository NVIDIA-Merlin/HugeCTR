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

#include "../embedding_table.hpp"
#include "HugeCTR/core/registry.hpp"

namespace embedding {

struct ReductionIndices;
struct ModelCommBuffer;

struct PartialReduceResult {
  Tensor partial_wgrad;
  Tensor partial_keys;
  Tensor partial_ev_length;
  Tensor partial_dst_offset_array;

  Tensor partial_wgrad_new;
  Tensor partial_ev_length_new;
  Tensor partial_dst_id_array_new;
  size_t max_input_num;
};

class LocalReduce {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  KernelParams kernel_params_;
  PartialReduceResult partial_reduce_result_;

 public:
  void init(std::shared_ptr<CoreResourceManager> core, const KernelParams &kernel_params,
            int max_ev_size, size_t max_input_num);

  void local_reduce(const ReductionIndices &reduction_indices, const ModelCommBuffer &src_buffer,
                    Wgrad &wgrad, int batch_size);

  void local_reduce(const ReductionIndices &reduction_indices, const EmbeddingOutput &src_buffer,
                    Wgrad &wgrad, const Tensor &local_lookup_ids, int num_lookup,
                    int num_global_lookup, int batch_size);
};
}  // namespace embedding
