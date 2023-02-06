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

#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/weighted_model_forward.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace embedding {
using HugeCTR::CudaDeviceContext;

WeightedModelForward::WeightedModelForward(std::shared_ptr<CoreResourceManager> core, int num_gpus,
                                           const std::vector<int> &local_embedding_list)
    : core_(core), num_gpus_(num_gpus), num_local_embedding_(local_embedding_list.size()) {}

void WeightedModelForward::compute(const TensorList &mp_ev, const Tensor &model_offset,
                                   TensorList &model_comm_buffer,
                                   const Tensor &d_local_ev_size_list,
                                   const Tensor &d_local_ev_size_offset, int batch_size,
                                   int max_ev_size, const Tensor &sp_weight) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();
  auto stream = core_->get_local_gpu()->get_stream();

  if (num_local_embedding_ > 0) {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(model_comm_buffer.dtype().type(), emb_t, [&] {
      const uint32_t *model_offset_ptr = model_offset.get<uint32_t>();
      const int *d_local_ev_size_list_ptr = d_local_ev_size_list.get<int>();
      const int *d_local_ev_size_offset_ptr = d_local_ev_size_offset.get<int>();
      const float **mp_ev_ptr = mp_ev.get<float>();
      emb_t **model_comm_buffer_ptr = model_comm_buffer.get<emb_t>();
      const float *sp_weight_ptr = sp_weight.get<float>();

      auto multi_to_one_desc = make_MultiToOneWeight<float, emb_t>(
          batch_size * num_local_embedding_, [=] __device__(int i) { return model_offset_ptr[i]; },
          [=] __device__(int i) { return 1.0f; },
          [=] __device__(int i) {
            int i_lookup = i / batch_size;
            return d_local_ev_size_list_ptr[i_lookup];
          },
          [=] __device__(int i) { return mp_ev_ptr[i]; },
          [=] __device__(int i) {
            int i_lookup = i / batch_size;
            int batch_id = i % batch_size;
            int gpu_id = batch_id / batch_size_per_gpu;
            int ev_size =
                d_local_ev_size_offset_ptr[i_lookup + 1] - d_local_ev_size_offset_ptr[i_lookup];
            int local_batch_id = batch_id % batch_size_per_gpu;
            return model_comm_buffer_ptr[gpu_id] +
                   batch_size_per_gpu * d_local_ev_size_offset_ptr[i_lookup] +
                   local_batch_id * ev_size;
          },
          [=] __device__(int i) { return sp_weight_ptr[i]; });
      copy_multi_to_one_weight(multi_to_one_desc, max_ev_size, stream);
    });
  }
}
}  // namespace embedding
