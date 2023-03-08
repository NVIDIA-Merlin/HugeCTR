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
#include "HugeCTR/include/utils.cuh"
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "hier_model_forward.hpp"
#include "model_forward.hpp"

namespace embedding {
using HugeCTR::CudaDeviceContext;

embedding::IntraModelCommBufferAttr::IntraModelCommBufferAttr(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : type(ebc_param.emb_type) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  h_lookup_ids_in_current_node.clear();
  h_id_to_ev_size_in_current_node.clear();
  h_id_to_ev_start_indices_in_current_node.clear();

  int num_local_gpus = static_cast<int>(core->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core->get_global_gpu_count());
  HCTR_CHECK(num_global_gpus % num_local_gpus == 0);

  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    int global_gpu_id = core->get_gpu_global_id_from_local_id(local_gpu_id);

    std::vector<int> local_lookup_ids;
    std::vector<int> local_id_to_ev_size;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      if (!ebc_param.has_table_shard(global_gpu_id, grouped_id, lookup_id)) continue;

      local_lookup_ids.push_back(lookup_id);

      int ev_size = ebc_param.lookup_params[lookup_id].ev_size;
      local_id_to_ev_size.push_back(ev_size);
    }
    std::vector<int> local_id_to_ev_start_indices{0};
    std::partial_sum(local_id_to_ev_size.begin(), local_id_to_ev_size.end(),
                     std::back_inserter(local_id_to_ev_start_indices));

    h_lookup_ids_in_current_node.push_back(local_lookup_ids);
    h_id_to_ev_size_in_current_node.push_back(local_id_to_ev_size);
    h_id_to_ev_start_indices_in_current_node.push_back(local_id_to_ev_start_indices);
  }

  {
    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    id_to_ev_size_in_current_node_list.clear();
    id_to_ev_start_indices_in_current_node_list.clear();
    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      id_to_ev_size_in_current_node_list.emplace_back(
          params.shape({static_cast<int64_t>(h_id_to_ev_size_in_current_node[local_gpu_id].size())})
              .data_type(core23::ScalarType::Int32));
      id_to_ev_start_indices_in_current_node_list.emplace_back(
          params
              .shape({static_cast<int64_t>(
                  h_id_to_ev_start_indices_in_current_node[local_gpu_id].size())})
              .data_type(core23::ScalarType::Int32));
    }

    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      core23::copy_sync(id_to_ev_size_in_current_node_list[local_gpu_id],
                        h_id_to_ev_size_in_current_node[local_gpu_id]);
      core23::copy_sync(id_to_ev_start_indices_in_current_node_list[local_gpu_id],
                        h_id_to_ev_start_indices_in_current_node[local_gpu_id]);
    }

    id_to_ev_size_in_current_node =
        core23::init_tensor_list<int>(id_to_ev_size_in_current_node_list, device.index());
    id_to_ev_start_indices_in_current_node =
        core23::init_tensor_list<int>(id_to_ev_start_indices_in_current_node_list, device.index());
  }

  {
    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);
    const auto &h_id_to_ev_size_in_current_gpu =
        h_id_to_ev_size_in_current_node[core->get_local_gpu_id()];
    const auto &h_id_to_ev_start_indices_in_current_gpu =
        h_id_to_ev_start_indices_in_current_node[core->get_local_gpu_id()];

    num_local_lookup = static_cast<int>(h_id_to_ev_size_in_current_gpu.size());

    id_to_ev_size_in_current_gpu =
        core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_ev_size_in_current_gpu.size())})
                           .data_type(core23::ScalarType::Int32));
    id_to_ev_start_indices_in_current_gpu = core23::Tensor(
        params.shape({static_cast<int64_t>(h_id_to_ev_start_indices_in_current_gpu.size())})
            .data_type(core23::ScalarType::Int32));

    core23::copy_sync(id_to_ev_size_in_current_gpu, h_id_to_ev_size_in_current_gpu);
    core23::copy_sync(id_to_ev_start_indices_in_current_gpu,
                      h_id_to_ev_start_indices_in_current_gpu);

    max_ev_size = h_id_to_ev_size_in_current_gpu.empty()
                      ? 0
                      : *std::max_element(h_id_to_ev_size_in_current_gpu.begin(),
                                          h_id_to_ev_size_in_current_gpu.end());
  }
}

IntraModelCommBuffer::IntraModelCommBuffer(std::shared_ptr<CoreResourceManager> core,
                                           const IntraModelCommBufferAttr &attr, int batch_size)
    : attr(attr) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());

  local_datas.clear();
  int num_local_gpus = static_cast<int>(core->get_local_gpu_count());
  int batch_size_per_rail = batch_size / num_local_gpus;

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    local_datas.emplace_back(
        params
            .shape({batch_size_per_rail,
                    attr.h_id_to_ev_start_indices_in_current_node[local_gpu_id].back()})
            .data_type(attr.type));
  }

  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(attr.type.type(), emb_t, [&] {
    local_datas_device_view = core23::init_tensor_list<emb_t>(this->local_datas, device.index());
  });
}

void collective_init_peer_buffer(const std::vector<std::shared_ptr<CoreResourceManager>> &cores,
                                 std::vector<IntraModelCommBuffer *> &intra_model_comm_buffers) {
  int num_local_gpus = static_cast<int>(cores.size());

  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
      intra_model_comm_buffers[0]->attr.type.type(), emb_t, [&] {
        std::vector<emb_t **> peer_data_vec;
        for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
          peer_data_vec.push_back(
              (emb_t **)intra_model_comm_buffers[local_gpu_id]->local_datas_device_view.data());
        }

        for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
          HugeCTR::CudaDeviceContext context(cores[local_gpu_id]->get_device_id());

          core23::Device device(core23::DeviceType::GPU, cores[local_gpu_id]->get_device_id());
          core23::TensorParams params = core23::TensorParams().device(device);

          intra_model_comm_buffers[local_gpu_id]->peer_data = core23::Tensor(
              params.shape({static_cast<int64_t>(peer_data_vec.size()), sizeof(emb_t **)})
                  .data_type({core23::ScalarType::Char}));

          HCTR_LIB_THROW(cudaMemcpy(intra_model_comm_buffers[local_gpu_id]->peer_data.data(),
                                    peer_data_vec.data(), peer_data_vec.size() * sizeof(emb_t ***),
                                    cudaMemcpyHostToDevice));
        }
      });
}

IntraModelReductionBufferAttr::IntraModelReductionBufferAttr(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id, const std::vector<std::vector<int>> &h_lookup_ids_in_current_node)
    : type(ebc_param.emb_type) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel &&
                      group_params.comm_strategy == CommunicationStrategy::Hierarchical,
                  "IntraModelReductionBufferAttr must be initialized by ModelParallel & "
                  "Hierarchical communication");

  int num_local_gpus = static_cast<int>(core->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core->get_global_gpu_count());
  HCTR_CHECK(num_global_gpus % num_local_gpus == 0);

  indices.init(core, h_lookup_ids_in_current_node);

  h_inter_id_to_ev_size.clear();
  for (int lookup_id : indices.h_network_dst_lookup_ids) {
    h_inter_id_to_ev_size.push_back(ebc_param.lookup_params[lookup_id].ev_size);
  }

  h_inter_id_to_ev_start_indices = {0};
  std::partial_sum(h_inter_id_to_ev_size.begin(), h_inter_id_to_ev_size.end(),
                   std::back_inserter(h_inter_id_to_ev_start_indices));

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  id_to_ev_size = core23::Tensor(params.shape({static_cast<int64_t>(h_inter_id_to_ev_size.size())})
                                     .data_type(core23::ScalarType::Int32));
  id_to_ev_start_indices =
      core23::Tensor(params.shape({static_cast<int64_t>(h_inter_id_to_ev_start_indices.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(id_to_ev_size, h_inter_id_to_ev_size);
  core23::copy_sync(id_to_ev_start_indices, h_inter_id_to_ev_start_indices);

  this->max_ev_size =
      h_inter_id_to_ev_size.empty()
          ? 0
          : *std::max_element(h_inter_id_to_ev_size.begin(), h_inter_id_to_ev_size.end());
}

IntraModelReductionBuffer::IntraModelReductionBuffer(std::shared_ptr<CoreResourceManager> core,
                                                     const IntraModelReductionBufferAttr &attr,
                                                     int batch_size)
    : attr(attr) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());

  data_list.clear();

  int num_local_gpus = static_cast<int>(core->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core->get_global_gpu_count());
  HCTR_CHECK(num_global_gpus % num_local_gpus == 0);

  int num_nodes = num_global_gpus / num_local_gpus;

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    data_list.emplace_back(
        params.shape({batch_size / num_global_gpus, attr.h_inter_id_to_ev_start_indices.back()})
            .data_type(attr.type));
  }

  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(attr.type.type(), emb_t, [&] {
    data = core23::init_tensor_list<emb_t>(this->data_list, core->get_device_id());
  });
}

template <typename emb_t, typename offset_t>
struct IntraModelForwardMultiToOneDesc {
  using SrcT = float;
  using DstT = emb_t;

  HOST_DEVICE_INLINE int get_offset(int i) { return bucket_range_ptr[i]; }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    int i_lookup = i / batch_size;
    return id_to_ev_size_ptr[i_lookup];
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) { return 1; }
  HOST_DEVICE_INLINE const SrcT *get_src_ptr(int i) { return evs_ptr[i]; }
  HOST_DEVICE_INLINE DstT *get_dst_ptr(int i) {
    int i_lookup = i / batch_size;
    int batch_id = i % batch_size;
    int local_batch_id =
        (batch_id / batch_size_per_node) * batch_size_per_gpu + batch_id % batch_size_per_gpu;
    int dst_gpu_id = (batch_id / batch_size_per_gpu) % num_local_gpus;
    int ev_size = id_to_ev_size_ptr[i_lookup];

    return peer_data_ptr[dst_gpu_id][src_gpu_id] +
           batch_size_per_rail * dst_id_to_ev_start_indices_ptr[i_lookup] +
           local_batch_id * ev_size;
  }

  int num_vec_;

  const offset_t *__restrict__ bucket_range_ptr;
  const float **__restrict__ evs_ptr;
  emb_t ***peer_data_ptr;

  const int *__restrict__ id_to_ev_size_ptr;
  const int *__restrict__ dst_id_to_ev_start_indices_ptr;

  int batch_size;
  int batch_size_per_rail;
  int batch_size_per_node;
  int batch_size_per_gpu;

  int src_gpu_id;
  int num_local_gpus;
};

void IntraModelForward::intra_forward(const core23::Tensor &evs, const core23::Tensor &bucket_range,
                                      IntraModelCommBuffer &intra_model_comm_buffer,
                                      int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int local_gpu_id = core_->get_local_gpu_id();
  int num_local_lookup = intra_model_comm_buffer.attr.num_local_lookup;
  int num_local_gpus = static_cast<int>(core_->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core_->get_global_gpu_count());
  int batch_size_per_rail = batch_size / num_local_gpus;
  int batch_size_per_node = batch_size / (num_global_gpus / num_local_gpus);
  int batch_size_per_gpu = batch_size / num_global_gpus;
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(intra_model_comm_buffer.attr.type.type(), emb_t, [&] {
      auto peer_data_ptr = reinterpret_cast<emb_t ***>(intra_model_comm_buffer.peer_data.data());

      using CopyDesc = IntraModelForwardMultiToOneDesc<emb_t, offset_t>;
      CopyDesc multi_to_one_desc{
          batch_size * num_local_lookup,
          bucket_range.data<offset_t>(),
          (const float **)evs.data(),
          peer_data_ptr,
          intra_model_comm_buffer.attr.id_to_ev_size_in_current_gpu.data<int>(),
          intra_model_comm_buffer.attr.id_to_ev_start_indices_in_current_gpu.data<int>(),
          batch_size,
          batch_size_per_rail,
          batch_size_per_node,
          batch_size_per_gpu,
          local_gpu_id,
          num_local_gpus,
      };
      copy_multi_to_one(multi_to_one_desc, intra_model_comm_buffer.attr.max_ev_size, stream);
    });
  });
}

//
// batch_size_per_node * num_local_lookup -> num_node * batch_size_per_gpu *
// num_network_dst_lookup_ids
template <typename emb_t>
struct IntraModelDstReductionMultiToOneDesc {
  using SrcT = emb_t;
  using DstT = emb_t;

  HOST_DEVICE_INLINE int get_offset(int i) {
    int bid = i / num_network_dst_lookup_ids;
    int lookup_id = i % num_network_dst_lookup_ids;
    return bid * network_offsets_ptr[num_network_dst_lookup_ids] + network_offsets_ptr[lookup_id];
  }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    return dst_id_to_ev_size_ptr[i % num_network_dst_lookup_ids];
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) { return 1; }
  HOST_DEVICE_INLINE const SrcT *get_src_ptr(int i) {
    int id = i % network_offsets_ptr[num_network_dst_lookup_ids];
    int network_gpu_id = network_gpu_ids_ptr[id];
    int network_id = network_ids_ptr[id];

    int ev_offset = src_id_to_ev_start_indices_in_current_node_ptr[network_gpu_id][network_id] *
                    batch_size_per_rail;
    int ev_size = src_id_to_ev_size_in_current_node_ptr[network_gpu_id][network_id];

    int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
    return model_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
  }
  HOST_DEVICE_INLINE DstT *get_dst_ptr(int i) {
    int bid = i / num_network_dst_lookup_ids;
    int node_id = bid / batch_size_per_gpu;
    int local_bid = bid % batch_size_per_gpu;

    int i_lookup = i % num_network_dst_lookup_ids;
    int ev_offset = batch_size_per_gpu * dst_id_to_ev_start_indices_ptr[i_lookup];
    int ev_size = dst_id_to_ev_size_ptr[i_lookup];

    return reduction_buffer_ptr[node_id] + ev_offset + local_bid * ev_size;
  }

  int num_vec_;

  const emb_t **__restrict__ model_comm_buffer_ptr;
  emb_t **__restrict__ reduction_buffer_ptr;

  const int *__restrict__ network_ids_ptr;
  const int *__restrict__ network_gpu_ids_ptr;
  const int *__restrict__ network_offsets_ptr;

  const int **__restrict__ src_id_to_ev_size_in_current_node_ptr;
  const int **__restrict__ src_id_to_ev_start_indices_in_current_node_ptr;

  const int *__restrict__ dst_id_to_ev_size_ptr;
  const int *__restrict__ dst_id_to_ev_start_indices_ptr;

  int batch_size_per_gpu;
  int batch_size_per_rail;
  int num_network_dst_lookup_ids;
};

void IntraModelForward::dst_reduction(const IntraModelCommBuffer &intra_model_comm_buffer,
                                      IntraModelReductionBuffer &reduction_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  HCTR_CHECK(intra_model_comm_buffer.attr.type == reduction_buffer.attr.type);
  auto emb_type = intra_model_comm_buffer.attr.type;

  int num_local_gpus = static_cast<int>(core_->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core_->get_global_gpu_count());
  int num_node = num_global_gpus / num_local_gpus;

  int batch_size_per_gpu = batch_size / num_global_gpus;
  int batch_size_per_rail = batch_size / num_local_gpus;
  int num_network_dst_lookup_ids =
      reduction_buffer.attr.indices.network_dst_lookup_ids.num_elements();
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(emb_type.type(), emb_t, [&] {
    auto *peer_data_ptr = reinterpret_cast<emb_t ***>(intra_model_comm_buffer.peer_data.data());

    using CopyDesc = IntraModelDstReductionMultiToOneDesc<emb_t>;
    CopyDesc multi_to_one_desc{
        batch_size_per_gpu * num_node * num_network_dst_lookup_ids,
        (const emb_t **)intra_model_comm_buffer.local_datas_device_view.data(),
        (emb_t **)reduction_buffer.data.data(),
        reduction_buffer.attr.indices.network_ids.data<int>(),
        reduction_buffer.attr.indices.network_gpu_ids.data<int>(),
        reduction_buffer.attr.indices.network_offsets.data<int>(),
        (const int **)intra_model_comm_buffer.attr.id_to_ev_size_in_current_node.data(),
        (const int **)intra_model_comm_buffer.attr.id_to_ev_start_indices_in_current_node.data(),
        reduction_buffer.attr.id_to_ev_size.data<int>(),
        reduction_buffer.attr.id_to_ev_start_indices.data<int>(),
        batch_size_per_gpu,
        batch_size_per_rail,
        num_network_dst_lookup_ids,
    };
    copy_multi_to_one(multi_to_one_desc, intra_model_comm_buffer.attr.max_ev_size, stream);
  });
}
}  // namespace embedding
