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

#include <cub/cub.cuh>
#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/mp_index_calculation.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace embedding {

// Cautions:num_key is host value ,should we move to device value? and support cuda graph
template <typename key_t, typename offset_t>
__global__ void dense_lookup_wgrad_attr_calculation_kernel(key_t* keys_input, int* table_ids_input,
                                                           offset_t* key_range_per_table,
                                                           size_t num_id_space_offset,
                                                           size_t num_key_input, int ev_size,
                                                           key_t* keys, uint32_t* ev_start_indices,
                                                           int* table_ids, uint64_t* num_key) {
  if (threadIdx.x + blockIdx.x == 0) {
    num_key[0] = (uint64_t)num_key_input;
  }
  CUDA_1D_KERNEL_LOOP(i, num_key_input) {
    int64_t table_idx =
        bs_upper_bound_sub_one(key_range_per_table, num_id_space_offset, (offset_t)i);
    keys[i] = keys_input[i];
    table_ids[i] = table_ids_input[table_idx];
    ev_start_indices[i] = i * ev_size;
  }
}

void MPLocalReduceIndexCalculation::init(
    std::shared_ptr<CoreResourceManager> core,
    const LocalReduceIndexCalculation& local_reduce_index_calculation,
    const SortKeyAndSrcIdOp& sort_op, const CalDstIds& cal_dst_ids,
    const SegmentdUnique& segmented_unique, const CalDstOffsetMP& cal_dst_offset_mp) {
  core_ = core;
  local_reduce_index_calculation_ = local_reduce_index_calculation;
  sort_op_ = sort_op;
  cal_dst_ids_ = cal_dst_ids;
  segmented_unique_ = segmented_unique;
  cal_dst_offset_mp_ = cal_dst_offset_mp;
}

void MPLocalReduceIndexCalculation::cal_for_sparse_input(const EmbeddingInput& embedding_input,
                                                         ReductionIndices& reduction_indices,
                                                         Wgrad& wgrad, int batch_size) {
  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, sort_op_, segmented_unique_,
                                                       reduction_indices, wgrad, batch_size);

  auto stream = core_->get_local_gpu()->get_stream();
  if (!wgrad.attr.is_same_ev_size && embedding_input.h_num_keys != 0) {
    cal_dst_offset_mp_(wgrad.table_ids, wgrad.attr.table_id_to_ev_size, wgrad.num_unique_keys,
                       wgrad.ev_start_indices, core_, stream);
  }
}

void DenseMPLocalReduceIndexCalculation::init(std::shared_ptr<CoreResourceManager> core) {
  core_ = core;
}

void DenseMPLocalReduceIndexCalculation::cal_for_dense_input(
    const EmbeddingInput& embedding_input, DenseReductionIndices& reduction_indices, Wgrad& wgrad,
    int ev_size) {
  auto stream = core_->get_local_gpu()->get_stream();
  auto key_type = embedding_input.keys.data_type();
  auto offset_type = embedding_input.dense_compression_input.num_keys_per_table_offset.data_type();
  reduction_indices.reverse_key_num = embedding_input.dense_compression_input
                                          .model_parallel_compression_input.num_model_reverse_idx;
  reduction_indices.model_reverse_idx =
      &embedding_input.dense_compression_input.model_parallel_compression_input.model_reverse_idx;
  reduction_indices.ev_size = ev_size;
  reduction_indices.num_valid_dst_tensor = embedding_input.h_num_keys;
  const int block_size = 256;
  const int grid_size = core_->get_kernel_param().num_sms *
                        core_->get_kernel_param().max_thread_per_block / block_size;
  size_t num_keys = embedding_input.h_num_keys;  // number of input unique keys
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(offset_type.type(), offset_t, [&] {
      dense_lookup_wgrad_attr_calculation_kernel<<<grid_size, block_size, 0, stream>>>(
          embedding_input.keys.data<key_t>(),
          embedding_input.dense_compression_input.table_ids.data<int>(),
          embedding_input.dense_compression_input.num_keys_per_table_offset.data<offset_t>(),
          embedding_input.dense_compression_input.num_keys_per_table_offset.num_elements(),
          num_keys, ev_size, wgrad.unique_keys.data<key_t>(),
          wgrad.ev_start_indices.data<uint32_t>(), wgrad.table_ids.data<int>(),
          wgrad.num_unique_keys.data<uint64_t>());
    });
  });
}

}  // namespace embedding
