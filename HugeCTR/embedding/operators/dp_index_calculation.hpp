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

#include <core23/registry.hpp>
#include <embedding/operators/index_calculation.hpp>

namespace embedding {
using core::CoreResourceManager;

struct DPKeySelector {
  int num_lookup_before_filter;
  core23::Tensor lookup_ids;
  int num_lookup_after_filter;

  int gpu_id;
  int num_gpus;

  int max_num_keys_before_filter;
  int max_num_keys_after_filter;
};

using DPIndexCalculation = IndexCalculation<DPKeySelector>;

struct DPLocalReduceIndexCalculation {
  std::shared_ptr<CoreResourceManager> core_;
  LocalReduceIndexCalculation local_reduce_index_calculation_;
  SortKeyAndSrcIdOp indices_sort_;
  CalDstIds cal_dst_ids_;
  SegmentdUnique segmented_unique_;

  void cal_for_sparse_indices(const EmbeddingInput& embedding_input,
                              const core23::Tensor& ev_start_indices_in_allreduce_buffer,
                              ReductionIndices& reduction_indices, Wgrad& wgrad,
                              int batch_size_per_gpu);

  void cal_for_dense_indices(const EmbeddingInput& embedding_input,
                             const core23::Tensor& ev_start_indices_in_allreduce_buffer,
                             ReductionIndices& reduction_indices, Wgrad& wgrad, int batch_size);
};

}  // namespace embedding
