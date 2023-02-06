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
#include <cub/cub.cuh>

#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "mp_index_calculation.hpp"

namespace embedding {
void MPLocalReduceIndexCalculation::init(
    std::shared_ptr<CoreResourceManager> core,
    const LocalReduceIndexCalculation& local_reduce_index_calculation,
    const SegmentedSortDevice& segmented_sort_device, const CalDstIds& cal_dst_ids,
    const SegmentdUnique& segmented_unique, const CalDstOffsetMP& cal_dst_offset_mp) {
  core_ = core;
  local_reduce_index_calculation_ = local_reduce_index_calculation;
  segmented_sort_device_ = segmented_sort_device;
  cal_dst_ids_ = cal_dst_ids;
  segmented_unique_ = segmented_unique;
  cal_dst_offset_mp_ = cal_dst_offset_mp;
}

void MPLocalReduceIndexCalculation::init(
    std::shared_ptr<CoreResourceManager> core,
    const LocalReduceIndexCalculation& local_reduce_index_calculation,
    const IndicesSort& indices_sort, const CalDstIds& cal_dst_ids,
    const SegmentdUnique& segmented_unique, const CalDstOffsetMP& cal_dst_offset_mp) {
  core_ = core;
  local_reduce_index_calculation_ = local_reduce_index_calculation;
  indices_sort_ = indices_sort;
  cal_dst_ids_ = cal_dst_ids;
  segmented_unique_ = segmented_unique;
  cal_dst_offset_mp_ = cal_dst_offset_mp;
}

void MPLocalReduceIndexCalculation::cal_for_sparse_input(const EmbeddingInput& embedding_input,
                                                         ReductionIndices& reduction_indices,
                                                         Wgrad& wgrad, int batch_size,
                                                         bool need_cal_unique_range) {
  auto cal_ev_start_indices_in_local_wgrad = [&](const WgradEvStartIndicesCalculationInput& input,
                                                 WgradEvStartIndicesCalculationOutput& output,
                                                 cudaStream_t stream) {
    cal_dst_offset_mp_(input.table_ids, input.table_id_to_ev_size, input.num_unique_keys,
                       output.ev_start_indices, stream);
  };
  if (!indices_sort_.table_id_to_global_start_indices.empty()) {
    local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, indices_sort_,
                                                         segmented_unique_, cal_dst_ids_,
                                                         reduction_indices, wgrad, batch_size);
    if (need_cal_unique_range) local_reduce_index_calculation_.cal_unique_key_table_range(wgrad);
    local_reduce_index_calculation_.cal_dst_ev_start(wgrad, cal_ev_start_indices_in_local_wgrad);
  } else {
    local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, segmented_sort_device_,
                                                         segmented_unique_, cal_dst_ids_,
                                                         reduction_indices, wgrad, batch_size);
    if (need_cal_unique_range) local_reduce_index_calculation_.cal_unique_key_table_range(wgrad);
    local_reduce_index_calculation_.cal_dst_ev_start(wgrad, cal_ev_start_indices_in_local_wgrad);
  }
}

}  // namespace embedding
