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
#include "HugeCTR/embedding/hier_model_parallel_embedding.hpp"

#include "HugeCTR/include/utils.hpp"

namespace embedding {

HierModelParallelEmbeddingMeta::HierModelParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : intra_model_buffer_attr(core, ebc_param, grouped_id) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(
      group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel &&
          ebc_param.comm_strategy_ == CommunicationStrategy::Hierarchical,
      "HierModelParallelEmbeddingMeta must be initialized by ModelParallel & Hierarchical comm");

  size_t num_global_gpus = core->get_global_gpu_count();

  int global_gpu_id = core->get_global_gpu_id();
  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_global_gpus,
                  "shard matrix should contain num_global_gpus row.");

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(global_gpu_id, grouped_id, lookup_id)) {
      continue;
    }

    int table_id = lookup_params[lookup_id].table_id;
    h_local_table_id_list_.push_back(table_id);
  }

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());
  {
    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    d_local_table_id_list_ =
        core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list_.size())})
                           .data_type(core23::ScalarType::Int32));
    core23::copy_sync(d_local_table_id_list_, h_local_table_id_list_);
  }

  model_buffer_attr.init(core, ebc_param, grouped_id);

  size_t num_local_gpus = core->get_local_gpu_count();
  HCTR_CHECK(num_global_gpus % num_local_gpus == 0);
  int num_nodes = num_global_gpus / num_local_gpus;
  intra_model_reduction_buffer_attr_in_all_nodes.clear();
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::vector<int>> h_lookup_ids_in_one_node;
    h_lookup_ids_in_one_node.resize(num_local_gpus);
    for (size_t local_ggpu_id = 0; local_ggpu_id < num_local_gpus; ++local_ggpu_id) {
      int global_ggpu_id = local_ggpu_id + node_id * num_local_gpus;
      for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
        if (!ebc_param.has_table_shard(global_ggpu_id, grouped_id, lookup_id)) continue;
        h_lookup_ids_in_one_node[local_ggpu_id].push_back(lookup_id);
      }
    }
    intra_model_reduction_buffer_attr_in_all_nodes.emplace_back(core, ebc_param, grouped_id,
                                                                h_lookup_ids_in_one_node);
  }

  h_lookup_ids_in_current_rail.clear();
  for (int nnode_id = 0; nnode_id < num_nodes; ++nnode_id) {
    h_lookup_ids_in_current_rail.push_back(
        intra_model_reduction_buffer_attr_in_all_nodes[nnode_id].indices.h_network_dst_lookup_ids);
  }
  hier_network_indices.init(core, h_lookup_ids_in_current_rail);
  hier_network_buffer_attr.init(core, ebc_param, grouped_id, h_lookup_ids_in_current_rail);

  wgrad_attr.init(core, ebc_param, grouped_id);

  output_attr.init(core, ebc_param);
  update_mutable_meta(core, ebc_param, grouped_id);
}

void HierModelParallelEmbeddingMeta::update_mutable_meta(std::shared_ptr<CoreResourceManager> core,
                                                         const EmbeddingCollectionParam &ebc_param,
                                                         size_t grouped_id) const {
  h_local_hotness_list_.clear();

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by ModelParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;

    int max_hotness = lookup_params[lookup_id].max_hotness;
    h_local_hotness_list_.push_back(max_hotness);
  }
  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);
  output_attr.update_mutable_data(core, ebc_param);
}

HierModelParallelEmbedding::HierModelParallelEmbedding(std::shared_ptr<CoreResourceManager> core,
                                                       const EmbeddingCollectionParam &params,
                                                       size_t grouped_id)
    : core_(core),
      meta_(core, params, grouped_id),
      intra_model_comm_buffer_(core, meta_.intra_model_buffer_attr, params.universal_batch_size),
      intra_reduction_buffer_(
          core,
          meta_.intra_model_reduction_buffer_attr_in_all_nodes[core->get_global_gpu_id() /
                                                               core->get_local_gpu_count()],
          params.universal_batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core->get_global_gpu_count();
  auto key_type = params.key_type;

  compress_offset_ = CompressOffset(core, meta_.num_local_lookup_ + 1);
  intra_model_forward_ = IntraModelForward{core};
  all2all_comm_ = NcclAll2AllComm(core);
  network_forward_ = NetworkForward(core, num_gpus);
  network_backward_ = NetworkBackward(core, num_gpus);
  intra_model_backward_ = IntraModelBackward{core};

  reduction_indices_.init(core, meta_.num_local_hotness_, params.universal_batch_size);
  LocalReduceIndexCalculation local_reduce_index_calculation{core,
                                                             meta_.wgrad_attr.num_lookup,
                                                             meta_.wgrad_attr.num_table,
                                                             meta_.num_local_hotness_,
                                                             params.universal_batch_size,
                                                             key_type};
  CalDstIds cal_dst_ids{core, meta_.num_local_hotness_, params.universal_batch_size};
  SegmentdUnique segmentd_unique{core, meta_.num_local_hotness_, params.universal_batch_size};
  CalDstOffsetMP cal_dst_offset_mp{core, meta_.num_local_hotness_, params.universal_batch_size};
  if (params.sort_strategy_ == SortStrategy::Radix) {
    IndicesSort indices_sort{core, meta_.num_local_hotness_, params.universal_batch_size, key_type};
    local_reduce_index_calculation_.init(core, local_reduce_index_calculation, indices_sort,
                                         cal_dst_ids, segmentd_unique, cal_dst_offset_mp);
  } else if (params.sort_strategy_ == SortStrategy::Segmented) {
    SegmentedSortDevice segmented_sort{core, meta_.num_local_hotness_, params.universal_batch_size,
                                       meta_.wgrad_attr.num_table, key_type};
    local_reduce_index_calculation_.init(core, local_reduce_index_calculation, segmented_sort,
                                         cal_dst_ids, segmentd_unique, cal_dst_offset_mp);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "sort strategy not supported.");
  }

  local_reduce_.init(core, meta_.output_attr.max_ev_size,
                     meta_.num_local_hotness_ * params.universal_batch_size);

  embedding_vec_ = core23::init_tensor_list<float>(
      params.universal_batch_size * meta_.num_local_hotness_, core->get_device_id());

  model_comm_buffer_.init(core, meta_.model_buffer_attr, params.universal_batch_size);
  network_buffer_.init(core, meta_.hier_network_buffer_attr, params.universal_batch_size);
}

void HierModelParallelEmbedding::forward_per_gpu(const EmbeddingInput &embedding_input,
                                                 ILookup *embedding_table,
                                                 EmbeddingOutput &embedding_output,
                                                 int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  core23::Tensor num_key_per_lookup_offset;
  compress_offset_.compute(embedding_input.bucket_range, batch_size, &num_key_per_lookup_offset);

  embedding_table->lookup(embedding_input.keys, embedding_input.h_num_keys,
                          num_key_per_lookup_offset, meta_.num_local_lookup_ + 1,
                          meta_.d_local_table_id_list_, embedding_vec_);

  intra_model_forward_.intra_forward(embedding_vec_, embedding_input.bucket_range,
                                     intra_model_comm_buffer_, batch_size);
  gpu_barrier_->sync_all_gpus(core_->get_local_gpu()->get_stream(), core_->get_local_gpu_id());

  intra_model_forward_.dst_reduction(intra_model_comm_buffer_, intra_reduction_buffer_, batch_size);

  all2all_comm_.hier_communicate(intra_reduction_buffer_.data_list, network_buffer_.data_list);
  network_forward_.compute(embedding_input.fullbatch_bucket_range, network_buffer_,
                           meta_.hier_network_indices, embedding_output, batch_size);
}

void HierModelParallelEmbedding::backward_per_gpu(const EmbeddingInput &embedding_input,
                                                  const EmbeddingOutput &top_grad, Wgrad &wgrad,
                                                  int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, reduction_indices_, wgrad,
                                                       batch_size);

  network_backward_.compute(embedding_input.fullbatch_bucket_range, top_grad,
                            meta_.hier_network_indices, network_buffer_, batch_size);

  all2all_comm_.hier_communicate(network_buffer_.data_list, intra_reduction_buffer_.data_list);
  intra_model_backward_.backward(intra_model_comm_buffer_.attr, intra_reduction_buffer_,
                                 model_comm_buffer_, batch_size);
  gpu_barrier_->sync_all_gpus(core_->get_local_gpu()->get_stream(), core_->get_local_gpu_id());

  local_reduce_.local_reduce(reduction_indices_, model_comm_buffer_, wgrad, batch_size);
}
}  // namespace embedding
