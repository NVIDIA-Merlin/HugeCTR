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

#include <HugeCTR/embedding/operators/communication.hpp>
#include <cub/cub.cuh>
#include <embedding/data_distributor/data_compression_operators.cuh>
#include <embedding/operators/dp_index_calculation.hpp>
#include <utils.cuh>

namespace embedding {
namespace {

template <typename key_t>
__global__ void cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel(
    const key_t* unique_indices, int num_elements,
    const uint32_t* ev_start_indices_in_allreduce_buffer, const size_t* num_unique_key,
    uint32_t* ev_start_indices_for_local_reduce) {
  uint32_t num_keys = static_cast<uint32_t>(*num_unique_key);
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_elements) {
    if (i >= num_keys) {
      ev_start_indices_for_local_reduce[i] = 0;
      continue;
    }
    uint32_t idx = i;

    int idx_in_allreduce_buffer = static_cast<int>(unique_indices[idx]);

    ev_start_indices_for_local_reduce[i] =
        ev_start_indices_in_allreduce_buffer[idx_in_allreduce_buffer];
  }
}
}  // namespace

void DPLocalReduceIndexCalculation::cal_for_sparse_indices(
    const EmbeddingInput& embedding_input,
    const core23::Tensor& ev_start_indices_in_allreduce_buffer, ReductionIndices& reduction_indices,
    Wgrad& wgrad, int batch_size_per_gpu) {
  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, indices_sort_,
                                                       segmented_unique_, reduction_indices, wgrad,
                                                       batch_size_per_gpu);
  if (embedding_input.h_num_keys == 0) return;
  auto key_type = wgrad.unique_keys.data_type();
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel<<<144 * 8, 256, 0, stream>>>(
        wgrad.unique_keys.data<key_t>(), wgrad.unique_keys.num_elements(),
        ev_start_indices_in_allreduce_buffer.data<uint32_t>(), wgrad.num_unique_keys.data<size_t>(),
        wgrad.ev_start_indices.data<uint32_t>());
  });
}

}  // namespace embedding
