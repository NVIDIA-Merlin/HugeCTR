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
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "hier_model_backward.hpp"
#include "hier_model_forward.hpp"

namespace embedding {

//
// src:  num_node * batch_size_per_gpu * num_network_dst_lookup_ids
// dst: batch_size_per_gpu * num_local_lookup
template <typename emb_t>
struct IntraModelBackwardOneToMultiDesc {
  using SrcT = emb_t;
  using DstT = emb_t;

  HOST_DEVICE_INLINE int get_offset(int i) {
    int bid = i / num_network_dst_lookup_ids;
    return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
           network_offsets_ptr[i % num_network_dst_lookup_ids];
  }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    return dst_id_to_ev_size_ptr[i % num_network_dst_lookup_ids];
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) { return 1; }
  HOST_DEVICE_INLINE const SrcT *get_src_ptr(int i) {
    int bid = i / num_network_dst_lookup_ids;
    int node_id = bid / batch_size_per_gpu;
    int local_bid = bid % batch_size_per_gpu;

    int i_lookup = i % num_network_dst_lookup_ids;
    int ev_offset = batch_size_per_gpu * dst_ev_start_indices_ptr[i_lookup];
    int ev_size = dst_id_to_ev_size_ptr[i_lookup];

    return reduction_buffer_ptr[node_id] + ev_offset + local_bid * ev_size;
  }
  HOST_DEVICE_INLINE DstT *get_dst_ptr(int i) {
    int id = i % network_offsets_ptr[num_network_dst_lookup_ids];
    int network_id = network_ids_ptr[id];
    int network_gpu_id = network_gpu_ids_ptr[id];

    int ev_offset = src_id_to_ev_start_indices_in_current_node_ptr[network_gpu_id][network_id] *
                    batch_size_per_gpu;
    int ev_size = src_id_to_ev_size_in_current_node_ptr[network_gpu_id][network_id];

    int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
    int local_bid = bid % batch_size_per_gpu;

    int dst_gpu_id = (bid / batch_size_per_gpu) * num_local_gpu + local_gpu_id;

    return peer_model_comm_buffer_ptr[network_gpu_id][dst_gpu_id] + ev_offset + local_bid * ev_size;
  }

  int num_vec_;

  const emb_t **__restrict__ reduction_buffer_ptr;
  emb_t ***peer_model_comm_buffer_ptr;

  const int *__restrict__ network_ids_ptr;
  const int *__restrict__ network_gpu_ids_ptr;
  const int *__restrict__ network_offsets_ptr;

  const int **__restrict__ src_id_to_ev_size_in_current_node_ptr;
  const int **__restrict__ src_id_to_ev_start_indices_in_current_node_ptr;

  const int *__restrict__ dst_id_to_ev_size_ptr;
  const int *__restrict__ dst_ev_start_indices_ptr;

  int batch_size_per_gpu;
  int local_gpu_id;
  int num_local_gpu;
  int num_network_dst_lookup_ids;
};

void IntraModelBackward::backward(const IntraModelCommBufferAttr &intra_model_comm_buffer_attr,
                                  const IntraModelReductionBuffer &reduction_buffer,
                                  ModelCommBuffer &model_comm_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  HCTR_CHECK(intra_model_comm_buffer_attr.type == reduction_buffer.attr.type);
  HCTR_CHECK(intra_model_comm_buffer_attr.type == model_comm_buffer.attr.type);
  auto emb_type = reduction_buffer.attr.type;

  int num_local_gpus = static_cast<int>(core_->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core_->get_global_gpu_count());
  int num_node = num_global_gpus / num_local_gpus;
  int local_gpu_id = core_->get_local_gpu_id();

  int batch_size_per_gpu = batch_size / num_global_gpus;
  int num_network_dst_lookup_ids =
      reduction_buffer.attr.indices.network_dst_lookup_ids.num_elements();
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(emb_type.type(), emb_t, [&] {
    auto peer_data_ptr = (emb_t ***)model_comm_buffer.peer_data.data();

    using CopyDesc = IntraModelBackwardOneToMultiDesc<emb_t>;
    CopyDesc one_to_multi_desc{
        num_node * batch_size_per_gpu * num_network_dst_lookup_ids,
        (const emb_t **)reduction_buffer.data.data(),
        peer_data_ptr,
        reduction_buffer.attr.indices.network_ids.data<int>(),
        reduction_buffer.attr.indices.network_gpu_ids.data<int>(),
        reduction_buffer.attr.indices.network_offsets.data<int>(),
        (const int **)intra_model_comm_buffer_attr.id_to_ev_size_in_current_node.data(),
        (const int **)intra_model_comm_buffer_attr.id_to_ev_start_indices_in_current_node.data(),
        reduction_buffer.attr.id_to_ev_size.data<int>(),
        reduction_buffer.attr.id_to_ev_start_indices.data<int>(),
        batch_size_per_gpu,
        local_gpu_id,
        num_local_gpus,
        num_network_dst_lookup_ids,
    };
    copy_one_to_multi(one_to_multi_desc, reduction_buffer.attr.max_ev_size, stream);
  });
}
}  // namespace embedding
