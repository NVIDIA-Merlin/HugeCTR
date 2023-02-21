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
#include <embedding/operators/weighted_network_forward.hpp>
#include <utils.hpp>

namespace embedding {
using namespace core;

WeightedNetworkForward::WeightedNetworkForward(std::shared_ptr<CoreResourceManager> core,
                                               int num_gpus)
    : core_(core), num_gpus_(num_gpus) {}

void WeightedNetworkForward::compute(
    const core23::Tensor& row_lengths, const core23::Tensor& d_combiner_list,
    const core23::Tensor& network_comm_buffer, const core23::Tensor& network_ids,
    const core23::Tensor& network_gpu_ids, const core23::Tensor& network_offsets,
    const core23::Tensor& network_dst_lookup_ids, const core23::Tensor& network_ev_sizes,
    const core23::Tensor& network_ev_offsets, core23::Tensor& output_buffer,
    const core23::Tensor& d_ev_size_offset, int batch_size, int max_ev_size,
    const core23::Tensor& sp_weight_sum) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(row_lengths.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();

        const offset_t** row_lengths_ptr = static_cast<const offset_t**>(row_lengths.data());
        const int* network_ids_ptr = network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_dst_lookup_ids.data<int>();
        const int** network_ev_sizes_ptr = static_cast<const int**>(network_ev_sizes.data());
        const int** network_ev_offsets_ptr = static_cast<const int**>(network_ev_offsets.data());
        const emb_t** network_comm_buffer_ptr =
            static_cast<const emb_t**>(network_comm_buffer.data());
        const int* d_ev_size_offset_ptr = d_ev_size_offset.data<int>();
        const char* combiner_ptr = d_combiner_list.data<char>();
        const float* sp_weight_ptr = sp_weight_sum.data<float>();
        dst_emb_t** output_buffer_ptr = static_cast<dst_emb_t**>(output_buffer.data());
        int num_network_dst_lookup_ids = network_dst_lookup_ids.num_elements();
        int gpu_id = core_->get_global_gpu_id();

        auto multi_to_one_desc = make_MultiToOneWeight<emb_t, dst_emb_t>(
            num_network_dst_lookup_ids * batch_size_per_gpu,
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = i % num_network_dst_lookup_ids;
              return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
                     network_offsets_ptr[lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                return sp_weight_ptr[lookup_id * batch_size_per_gpu + bid];
              } else {
                return 1.0f;
              }
            },
            [=] __device__(int i) {
              int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              return d_ev_size_offset_ptr[dst_lookup_id + 1] - d_ev_size_offset_ptr[dst_lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
              int id = i % network_offsets_ptr[num_network_dst_lookup_ids];

              int network_gpu_id = network_gpu_ids_ptr[id];
              int network_id = network_ids_ptr[id];
              int ev_offset =
                  network_ev_offsets_ptr[network_gpu_id][network_id] * batch_size_per_gpu;
              int ev_size = network_ev_sizes_ptr[network_gpu_id][network_id];

              return network_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
              return output_buffer_ptr[lookup_id] + bid * ev_size;
            },
            [=] __device__(int i) { return 1.0f; });
        copy_multi_to_one_weight(multi_to_one_desc, max_ev_size, stream);
      });
    });
  });
}
}  // namespace embedding
