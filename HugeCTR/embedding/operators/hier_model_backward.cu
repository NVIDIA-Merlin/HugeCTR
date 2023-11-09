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

static std::vector<int> get_local_gpu_lookup_ids_to_node_lookup_ids(
    const std::vector<int> &sorted_global_lookup_ids,
    const std::vector<int> &network_dst_lookup_ids, int gpu_id) {
  std::vector<int> local_gpu_lookup_ids_to_node_lookup_ids;
  local_gpu_lookup_ids_to_node_lookup_ids.clear();
  local_gpu_lookup_ids_to_node_lookup_ids.resize(sorted_global_lookup_ids.size());
  for (size_t i = 0; i < sorted_global_lookup_ids.size(); ++i) {
    auto iter = std::find(network_dst_lookup_ids.begin(), network_dst_lookup_ids.end(),
                          sorted_global_lookup_ids[i]);
    if (iter != network_dst_lookup_ids.end()) {
      auto idx = std::distance(network_dst_lookup_ids.begin(), iter);
      local_gpu_lookup_ids_to_node_lookup_ids[i] = (int)idx;
    } else {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                     "local gpu lookup id doesn't in node lookup ids ");
    }
  }
  return local_gpu_lookup_ids_to_node_lookup_ids;
}

static std::vector<int> get_global_lookup_ids(const EmbeddingCollectionParam &ebc_param,
                                              size_t grouped_id, int gpu_id) {
  std::vector<int> lookup_ids;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;
    lookup_ids.push_back(lookup_id);
  }

  return lookup_ids;
}

static std::vector<int> get_evsizes_in_local_gpu(const EmbeddingCollectionParam &ebc_param,
                                                 size_t grouped_id, int gpu_id) {
  const auto &lookup_params = ebc_param.lookup_params;

  std::vector<int> local_evsizes;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;
    local_evsizes.push_back(lookup_params[lookup_id].ev_size);
  }
  return local_evsizes;
}

void IntraModelBackwardAttr::init(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id, const std::vector<std::vector<int>> &h_lookup_ids_in_current_node) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &group_params = ebc_param.grouped_lookup_params[grouped_id];
  HCTR_CHECK_HINT(group_params.embedding_group_type == EmbeddingGroupType::SparseModelParallel &&
                      ebc_param.comm_strategy_ == CommunicationStrategy::Hierarchical,
                  "IntraModelReductionBufferAttr must be initialized by SparseModelParallel & "
                  "Hierarchical communication");

  int num_local_gpus = static_cast<int>(core->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core->get_global_gpu_count());
  HCTR_CHECK(num_global_gpus % num_local_gpus == 0);

  indices.init(core, h_lookup_ids_in_current_node);

  // calculatea local lookup id
  int gpu_id = core->get_global_gpu_id();
  this->h_global_lookup_ids_in_local_gpu = get_global_lookup_ids(ebc_param, grouped_id, gpu_id);
  this->h_evsizes_in_local_gpu = get_evsizes_in_local_gpu(ebc_param, grouped_id, gpu_id);
  this->h_local_gpu_lookup_ids_to_node_lookup_ids = get_local_gpu_lookup_ids_to_node_lookup_ids(
      this->h_global_lookup_ids_in_local_gpu, indices.h_network_dst_lookup_ids, gpu_id);

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  global_lookup_ids_in_local_gpu =
      core23::Tensor(params.shape({static_cast<int64_t>(h_global_lookup_ids_in_local_gpu.size())})
                         .data_type(core23::ScalarType::Int32));

  local_gpu_lookup_ids_to_node_lookup_ids = core23::Tensor(
      params.shape({static_cast<int64_t>(h_local_gpu_lookup_ids_to_node_lookup_ids.size())})
          .data_type(core23::ScalarType::Int32));

  evsizes_in_local_gpu =
      core23::Tensor(params.shape({static_cast<int64_t>(h_evsizes_in_local_gpu.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(global_lookup_ids_in_local_gpu, h_global_lookup_ids_in_local_gpu);
  core23::copy_sync(local_gpu_lookup_ids_to_node_lookup_ids,
                    h_local_gpu_lookup_ids_to_node_lookup_ids);
  core23::copy_sync(evsizes_in_local_gpu, h_evsizes_in_local_gpu);
}

// src:  num_node * batch_size_per_gpu * num_network_dst_lookup_ids
// dst: batch_size_per_gpu * num_local_lookup
template <typename emb_t, typename offset_t>
struct IntraModelBackwardOneToOneDesc {
  using SrcT = emb_t;
  using DstT = emb_t;

  HOST_DEVICE_INLINE bool need_copy(int i) {
    uint64_t bucket_num = (uint64_t)(bucket_range[i + 1] - bucket_range[i]);
    return (bucket_num != 0);
  }

  HOST_DEVICE_INLINE int get_vec_length(int i) {
    return local_gpu_lookup_id_to_evsize_ptr[i / batch_size];
  }

  HOST_DEVICE_INLINE const SrcT *get_src_ptr(int i) {
    // add a local lookup id to node lookup id
    int lookup_id = i / batch_size;
    int batch_id = i % batch_size;
    int gpu_id = batch_id / batch_size_per_gpu;
    int node_id = gpu_id / num_local_gpu;
    int target_id = gpu_id % num_local_gpu;
    int local_batch_id = batch_id % batch_size_per_gpu;

    int i_lookup = local_gpu_lookup_ids_to_node_lookup_ids[lookup_id];
    int ev_offset = batch_size_per_gpu * src_ev_start_indices_ptr[i_lookup];
    int ev_size = local_gpu_lookup_id_to_evsize_ptr[lookup_id];
    return reduction_buffer_ptr[target_id][node_id] + ev_offset + local_batch_id * ev_size;
  }
  HOST_DEVICE_INLINE DstT *get_dst_ptr(int i) {
    int lookup_id = i / batch_size;
    uint32_t bucket_id = lookup_id * batch_size + i % batch_size;
    int batch_id = bucket_id % batch_size;
    int gpu_id = batch_id / batch_size_per_gpu;
    int local_batch_id = batch_id % batch_size_per_gpu;
    int ev_size = local_gpu_lookup_id_to_evsize_ptr[lookup_id];

    return dst_ptr[gpu_id] + batch_size_per_gpu * dst_id_to_ev_start_indices_ptr[lookup_id] +
           local_batch_id * ev_size;
  }

  int num_vec_;

  const emb_t ***__restrict__ reduction_buffer_ptr;
  emb_t **dst_ptr;

  const offset_t *__restrict__ bucket_range;
  const int *__restrict__ local_gpu_lookup_ids_to_node_lookup_ids;
  const int *__restrict__ src_ev_start_indices_ptr;
  const int *__restrict__ local_gpu_lookup_id_to_evsize_ptr;
  const int *__restrict__ dst_id_to_ev_start_indices_ptr;

  int batch_size;
  int batch_size_per_gpu;
  int num_local_gpu;
};

void IntraModelBackward::backward(const IntraModelCommBufferAttr &intra_model_comm_buffer_attr,
                                  const IntraModelReductionBuffer &reduction_buffer,
                                  const EmbeddingInput &embedding_input,
                                  ModelCommBuffer &model_comm_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  HCTR_CHECK(intra_model_comm_buffer_attr.type == reduction_buffer.attr.type);
  HCTR_CHECK(intra_model_comm_buffer_attr.type == model_comm_buffer.attr.type);
  auto emb_type = reduction_buffer.attr.type;

  int num_local_gpus = static_cast<int>(core_->get_local_gpu_count());
  int num_global_gpus = static_cast<int>(core_->get_global_gpu_count());

  int batch_size_per_gpu = batch_size / num_global_gpus;
  int num_network_dst_lookup_ids =
      reduction_buffer.attr.indices.network_dst_lookup_ids.num_elements();

  auto stream = core_->get_local_gpu()->get_stream();

  if (num_network_dst_lookup_ids == 0) return;
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(emb_type.type(), emb_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(
        embedding_input.bucket_range.data_type().type(), offset_t, [&] {
          emb_t **dst_ptr = (emb_t **)model_comm_buffer.data.data();
          using CopyDesc = IntraModelBackwardOneToOneDesc<emb_t, offset_t>;
          CopyDesc one_to_one_desc{(int)(batch_size * attr.h_global_lookup_ids_in_local_gpu.size()),
                                   (const emb_t ***)reduction_buffer.peer_data.data(),
                                   dst_ptr,
                                   embedding_input.bucket_range.data<offset_t>(),
                                   attr.local_gpu_lookup_ids_to_node_lookup_ids.data<int>(),
                                   reduction_buffer.attr.id_to_ev_start_indices.data<int>(),
                                   attr.evsizes_in_local_gpu.data<int>(),
                                   model_comm_buffer.attr.id_to_ev_start_indices.data<int>(),
                                   batch_size,
                                   batch_size_per_gpu,
                                   num_local_gpus};
          copy_one_to_one(one_to_one_desc, reduction_buffer.attr.max_ev_size, stream);
        })
  });
}
}  // namespace embedding
