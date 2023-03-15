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

#include <embedding/common.hpp>
#include <embedding/operators/index_calculation.hpp>

namespace embedding {
using core::CoreResourceManager;

struct MPKeySelector {
  int num_lookup_before_filter;
  core23::Tensor lookup_ids;
  int num_lookup_after_filter;

  core23::Tensor shard_ids;
  core23::Tensor num_shards;

  int max_num_keys_before_filter;
  int max_num_keys_after_filter;
};

using ModelIndexCalculation = IndexCalculation<MPKeySelector>;

class MPLocalReduceIndexCalculation {
 private:
  std::shared_ptr<CoreResourceManager> core_;
  LocalReduceIndexCalculation local_reduce_index_calculation_;

  SortKeyAndSrcIdOp sort_op_;
  CalDstIds cal_dst_ids_;
  SegmentdUnique segmented_unique_;
  CalDstOffsetMP cal_dst_offset_mp_;

 public:
  void init(std::shared_ptr<CoreResourceManager> core,
            const LocalReduceIndexCalculation& local_reduce_index_calculation,
            const SortKeyAndSrcIdOp& sort_op, const CalDstIds& cal_dst_ids,
            const SegmentdUnique& segmented_unique, const CalDstOffsetMP& cal_dst_offset_mp);

  void cal_for_sparse_input(const EmbeddingInput& embedding_input,
                            ReductionIndices& reduction_indices, Wgrad& wgrad, int batch_size,
                            bool need_cal_unique_range = false);
};

}  // namespace embedding
