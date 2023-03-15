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

#include <embedding/model_parallel_embedding.hpp>
#include <utils.hpp>

namespace embedding {

UniformModelParallelEmbeddingMeta::UniformModelParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup), h_ev_size_offset_{0}, h_local_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by ModelParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int ev_size = lookup_params[lookup_id].ev_size;
    char combiner = static_cast<char>(lookup_params[lookup_id].combiner);

    h_ev_size_list_.push_back(ev_size);
    h_combiner_list_.push_back(combiner);
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) {
      continue;
    }
    int shard_id, num_shard;
    ebc_param.get_table_shard_id(gpu_id, table_id, &shard_id, &num_shard);

    h_local_shard_id_list_.push_back(shard_id);
    h_local_num_shards_list_.push_back(num_shard);
    h_local_table_id_list_.push_back(table_id);
    h_local_lookup_id_list_.push_back(lookup_id);
    h_local_ev_size_list_.push_back(ev_size);
  }
  std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                   std::back_inserter(h_ev_size_offset_));
  d_ev_size_offset_ = core23::Tensor(params.shape({static_cast<int64_t>(h_ev_size_offset_.size())})
                                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_ev_size_offset_, h_ev_size_offset_);

  max_ev_size_ = !h_ev_size_list_.empty()
                     ? *std::max_element(h_ev_size_list_.begin(), h_ev_size_list_.end())
                     : 0;

  // cudaDeviceProp device_prop;
  // cudaGetDeviceProperties(&device_prop, 0);
  // num_sms_ = device_prop.multiProcessorCount;
  // FIX: cudaGetDeviceProperties get ,cost too much time, need remove it to the start of program ,
  // not use per iteration,for now fix the num_sms_
  num_sms_ = 108;
  kernel_params.init();

  d_combiner_list_ = core23::Tensor(params.shape({static_cast<int64_t>(h_combiner_list_.size())})
                                        .data_type(core23::ScalarType::Char));
  core23::copy_sync(d_combiner_list_, h_combiner_list_);

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());

  d_local_shard_id_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_shard_id_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_shard_id_list_, h_local_shard_id_list_);

  d_local_num_shards_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_num_shards_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_num_shards_list_, h_local_num_shards_list_);

  d_local_table_id_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_table_id_list_, h_local_table_id_list_);

  d_local_lookup_id_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_lookup_id_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_lookup_id_list_, h_local_lookup_id_list_);

  d_local_ev_size_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_ev_size_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_ev_size_list_, h_local_ev_size_list_);

  std::partial_sum(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end(),
                   std::back_inserter(h_local_ev_size_offset_));
  d_local_ev_size_offset_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_ev_size_offset_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_ev_size_offset_, h_local_ev_size_offset_);

  model_buffer_attr.init(core, ebc_param, grouped_id);

  h_global_lookup_id_list_.resize(num_gpus);
  for (size_t ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      if (!ebc_param.has_table_shard(ggpu_id, grouped_id, lookup_id)) continue;

      h_global_lookup_id_list_[ggpu_id].push_back(lookup_id);
    }
  }
  network_indices.init(core, h_global_lookup_id_list_);
  network_buffer_attr.init(core, ebc_param, grouped_id, h_global_lookup_id_list_);
  wgrad_attr.init(core, ebc_param, grouped_id);

  update_mutable_meta(core, ebc_param, grouped_id);
}

void UniformModelParallelEmbeddingMeta::update_mutable_meta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id) const {
  h_hotness_list_.clear();
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

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int max_hotness = lookup_params[lookup_id].max_hotness;

    h_hotness_list_.push_back(max_hotness);
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }

    if (ebc_param.shard_matrix[gpu_id][table_id] == 0) {
      continue;
    }

    h_local_hotness_list_.push_back(max_hotness);
  }
  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);
  hotness_sum_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);
}

UniformModelParallelEmbedding::UniformModelParallelEmbedding(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    size_t grouped_id)
    : core_(core), meta_(core, params, grouped_id) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core->get_global_gpu_count();
  auto key_type = params.key_type;

  compress_offset_ = CompressOffset(core, meta_.num_local_lookup_ + 1);
  model_forward_ = ModelForward{core};
  all2all_comm_ = NcclAll2AllComm(core);
  network_forward_ = NetworkForward(core, num_gpus);
  network_backward_ = NetworkBackward(core, num_gpus);

  reduction_indices_.init(core, meta_.num_local_hotness_, params.universal_batch_size);
  LocalReduceIndexCalculation local_reduce_index_calculation{core,
                                                             meta_.wgrad_attr.num_lookup,
                                                             meta_.wgrad_attr.num_table,
                                                             meta_.num_local_hotness_,
                                                             params.universal_batch_size,
                                                             key_type};
  CalDstIds cal_dst_ids{core, meta_.num_local_hotness_, params.universal_batch_size};
  SegmentdUnique segmented_unique{core, meta_.num_local_hotness_, params.universal_batch_size};
  CalDstOffsetMP cal_dst_offset_mp{core, meta_.num_local_hotness_, params.universal_batch_size};
  if (params.sort_strategy_ == SortStrategy::Radix) {
    IndicesSort indices_sort{core, meta_.num_local_hotness_, params.universal_batch_size, key_type};
    local_reduce_index_calculation_.init(core, local_reduce_index_calculation, indices_sort,
                                         cal_dst_ids, segmented_unique, cal_dst_offset_mp);
  } else if (params.sort_strategy_ == SortStrategy::Segmented) {
    SegmentedSortDevice segmented_sort{core, meta_.num_local_hotness_, params.universal_batch_size,
                                       meta_.wgrad_attr.num_table, key_type};
    local_reduce_index_calculation_.init(core, local_reduce_index_calculation, segmented_sort,
                                         cal_dst_ids, segmented_unique, cal_dst_offset_mp);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "sort strategy not supported.");
  }

  local_reduce_.init(core, meta_.kernel_params, meta_.max_ev_size_,
                     meta_.num_local_hotness_ * params.universal_batch_size);

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams tensor_params = core23::TensorParams().device(device);

  embedding_vec_ = core23::init_tensor_list<float>(
      params.universal_batch_size * meta_.num_local_hotness_, core->get_device_id());

  model_comm_buffer_.init(core, meta_.model_buffer_attr, params.universal_batch_size);
  network_buffer_.init(core, meta_.network_buffer_attr, params.universal_batch_size);
}

void UniformModelParallelEmbedding::forward_per_gpu(const EmbeddingInput &embedding_input,
                                                    ILookup *embedding_table,
                                                    EmbeddingOutput &embedding_output,
                                                    int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  core23::Tensor num_key_per_lookup_offset;
  compress_offset_.compute(embedding_input.bucket_range, batch_size, &num_key_per_lookup_offset);

  embedding_table->lookup(embedding_input.keys, embedding_input.h_num_keys,
                          num_key_per_lookup_offset, meta_.num_local_lookup_ + 1,
                          meta_.d_local_table_id_list_, embedding_vec_);

  model_forward_.compute(embedding_vec_, embedding_input.bucket_range, model_comm_buffer_,
                         batch_size);

  all2all_comm_.communicate(model_comm_buffer_.data_list, network_buffer_.data_list);
  network_forward_.compute(embedding_input.fullbatch_bucket_range, network_buffer_,
                           meta_.network_indices, embedding_output, batch_size);
}

void UniformModelParallelEmbedding::backward_per_gpu(const EmbeddingInput &embedding_input,
                                                     const EmbeddingOutput &top_grad, Wgrad &wgrad,
                                                     int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, reduction_indices_, wgrad,
                                                       batch_size);

  network_backward_.compute(embedding_input.fullbatch_bucket_range, top_grad, meta_.network_indices,
                            network_buffer_, batch_size);

  all2all_comm_.communicate(network_buffer_.data_list, model_comm_buffer_.data_list);

  local_reduce_.local_reduce(reduction_indices_, model_comm_buffer_, wgrad, batch_size);
}
}  // namespace embedding
