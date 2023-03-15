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

#include <embedding/data_parallel_embedding.hpp>
#include <utils.hpp>

namespace embedding {

UniformDataParallelEmbeddingMeta::UniformDataParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam& ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup),
      h_ev_size_offset_{0},
      allreduce_strategy_(ebc_param.allreduce_strategy_) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  const auto& lookup_params = ebc_param.lookup_params;
  const auto& group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::DataParallel,
                  "UniformDataParallelEmbeddingMeta must be initialized by DataParallel");

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
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }
    HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                    "dp table must be shared on all gpus");
    h_local_combiner_list_.push_back(combiner);
    h_local_lookup_id_list_.push_back(lookup_id);
    h_local_ev_size_list_.push_back(ev_size);
    h_local_table_id_list_.push_back(table_id);
  }

  max_ev_size_ = h_ev_size_list_.size() > 0
                     ? *std::max_element(h_ev_size_list_.begin(), h_ev_size_list_.end())
                     : 0;
  std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                   std::back_inserter(h_ev_size_offset_));
  d_ev_size_offset_ = core23::Tensor(params.shape({static_cast<int64_t>(h_ev_size_offset_.size())})
                                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_ev_size_offset_, h_ev_size_offset_);

  d_combiner_list_ = core23::Tensor(params.shape({static_cast<int64_t>(h_combiner_list_.size())})
                                        .data_type(core23::ScalarType::Char));
  core23::copy_sync(d_combiner_list_, h_combiner_list_);

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());

  d_local_lookup_id_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_lookup_id_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_lookup_id_list_, h_local_lookup_id_list_);

  d_local_ev_size_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_ev_size_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_ev_size_list_, h_local_ev_size_list_);

  d_local_table_id_list_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list_.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(d_local_table_id_list_, h_local_table_id_list_);

  wgrad_attr.init(core, ebc_param, grouped_id);

  kernel_params.init();
  update_mutable_meta(core, ebc_param, grouped_id);
}

void UniformDataParallelEmbeddingMeta::update_mutable_meta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam& ebc_param,
    size_t grouped_id) const {
  h_hotness_list_.clear();
  h_local_hotness_list_.clear();

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto& lookup_params = ebc_param.lookup_params;
  const auto& group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::DataParallel,
                  "UniformDataParallelEmbeddingMeta must be initialized by DataParallel");

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
    HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                    "dp table must be shared on all gpus");
    h_local_hotness_list_.push_back(max_hotness);
  }
  num_hotness_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);

  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);
}

UniformDPEmbedding::UniformDPEmbedding(std::shared_ptr<CoreResourceManager> core,
                                       const EmbeddingCollectionParam& params, size_t grouped_id)
    : core_(core), meta_(core, params, grouped_id) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());

  int num_gpus = core->get_global_gpu_count();
  int universal_batch_size = params.universal_batch_size;
  auto key_type = params.key_type;
  // auto offset_type = params.offset_type;
  // auto emb_type = params.emb_type;

  // init op
  compress_offset_ = CompressOffset(core_, meta_.num_local_lookup_ + 1);

  dp_model_forward_ = DPModelForward(core_, num_gpus, meta_.num_lookup_, meta_.num_local_lookup_);

  allreduce_comm_ = NcclAllReduceInplaceComm(core_);
  if (std::find(meta_.h_local_combiner_list_.begin(), meta_.h_local_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) != meta_.h_local_combiner_list_.end()) {
    average_combiner_ = AverageCombiner(core, num_gpus, meta_.num_local_lookup_,
                                        meta_.h_ev_size_list_, params.universal_batch_size);
  }

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams tensor_params = core23::TensorParams().device(device);

  embedding_vec_ = core23::init_tensor_list<float>(universal_batch_size * meta_.num_local_hotness_,
                                                   core->get_device_id());

  reduction_indices_.init(core, meta_.num_local_hotness_, params.universal_batch_size / num_gpus);
  // init_indices local reduce buffer
  WgradInitializer{core, params, grouped_id, meta_.wgrad_attr}
      .init(local_reduce_indices_)
      .init_indices();
  LocalReduceIndexCalculation local_reduce_index_calculation{core,
                                                             meta_.wgrad_attr.num_lookup,
                                                             meta_.wgrad_attr.num_table,
                                                             meta_.num_local_hotness_,
                                                             params.universal_batch_size / num_gpus,
                                                             key_type};
  CalDstIds cal_dst_ids{core, meta_.num_local_hotness_, universal_batch_size / num_gpus};
  SegmentdUnique segmented_unique{core, meta_.num_local_hotness_, universal_batch_size / num_gpus};
  CalDstOffsetMP cal_dst_offset_mp{core, meta_.num_local_hotness_, universal_batch_size / num_gpus};
  SortKeyAndSrcIdOp sort_op;
  if (params.sort_strategy_ == SortStrategy::Radix) {
    IndicesSort indices_sort{core, meta_.num_local_hotness_, params.universal_batch_size / num_gpus,
                             key_type};
    sort_op = indices_sort;
  } else if (params.sort_strategy_ == SortStrategy::Segmented) {
    SegmentedSortDevice segmented_sort{core, meta_.num_local_hotness_,
                                       params.universal_batch_size / num_gpus,
                                       meta_.wgrad_attr.num_table, key_type};
    sort_op = segmented_sort;
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "sort strategy not supported.");
  }

  if (params.allreduce_strategy_ == AllreduceStrategy::Dense) {
    local_reduce_index_calculation_.dense_allreduce_index_calculation = {
        core, local_reduce_index_calculation, sort_op, cal_dst_ids, segmented_unique};
  } else if (params.allreduce_strategy_ == AllreduceStrategy::Sparse) {
    SparseAllreduceCalEVStartIndicesStorage sparse_allreduce_storage{
        core, meta_.wgrad_attr.num_table, meta_.num_local_hotness_,
        params.universal_batch_size / num_gpus, key_type};
    local_reduce_index_calculation_.sparse_allreduce_index_calculation = {
        core,
        local_reduce_index_calculation,
        sort_op,
        cal_dst_ids,
        segmented_unique,
        sparse_allreduce_storage};
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "allreduce strategy not supported.");
  }
  local_reduce_.init(core, meta_.kernel_params, meta_.max_ev_size_,
                     meta_.num_local_hotness_ * (params.universal_batch_size / num_gpus));
}

void UniformDPEmbedding::forward_per_gpu(const EmbeddingInput& embedding_input,
                                         ILookup* embedding_table,
                                         EmbeddingOutput& embedding_output, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  core23::Tensor num_key_per_lookup_offset;
  compress_offset_.compute(embedding_input.bucket_range, batch_size_per_gpu,
                           &num_key_per_lookup_offset);

  embedding_table->lookup(embedding_input.keys, embedding_input.h_num_keys,
                          num_key_per_lookup_offset, meta_.num_local_lookup_ + 1,
                          meta_.d_local_table_id_list_, embedding_vec_);
  dp_model_forward_.compute(embedding_vec_, embedding_input.bucket_range,
                            meta_.d_local_lookup_id_list_, embedding_output, batch_size_per_gpu);
}

void UniformDPEmbedding::backward_per_gpu_with_dense_allreduce(
    const EmbeddingInput& embedding_input, const embedding::EmbeddingOutput& top_grad,
    embedding::Wgrad& wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core_->get_global_gpu_count();
  Wgrad local_reduce_buffer{local_reduce_indices_.attr,
                            local_reduce_indices_.unique_keys,
                            local_reduce_indices_.num_unique_keys,
                            local_reduce_indices_.table_ids,
                            local_reduce_indices_.ev_start_indices,
                            local_reduce_indices_.table_range,
                            wgrad.data};
  local_reduce_index_calculation_.dense_allreduce_index_calculation.cal_for_sparse_indices(
      embedding_input, wgrad.ev_start_indices, reduction_indices_, local_reduce_buffer,
      batch_size / num_gpus);

  EmbeddingOutput top_grad_after_average_combiner = top_grad;
  if (std::find(meta_.h_local_combiner_list_.begin(), meta_.h_local_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) != meta_.h_local_combiner_list_.end()) {
    if (top_grad_after_average_combiner.attr.layout == EmbeddingLayout::FeatureMajor) {
      average_combiner_.compute_feature_major(
          embedding_input.fullbatch_bucket_range, top_grad.data, meta_.d_local_lookup_id_list_,
          meta_.d_combiner_list_, meta_.d_ev_size_offset_, batch_size, meta_.max_ev_size_);
    } else {
      average_combiner_.compute_batch_major(embedding_input.fullbatch_bucket_range, top_grad.data,
                                            meta_.d_local_lookup_id_list_, meta_.d_combiner_list_,
                                            meta_.d_ev_size_offset_, batch_size, meta_.max_ev_size_,
                                            meta_.num_lookup_);
    }

    top_grad_after_average_combiner.data = average_combiner_.float_emb_vec_;
    top_grad_after_average_combiner.attr.type = core23::ScalarType::Float;
  }
  local_reduce_.local_reduce(reduction_indices_, top_grad_after_average_combiner,
                             local_reduce_buffer, meta_.d_local_lookup_id_list_,
                             meta_.num_local_lookup_, meta_.num_lookup_, batch_size);

  allreduce_comm_.communicate(wgrad.data, wgrad.data.num_elements());
}

void UniformDPEmbedding::backward_per_gpu_with_sparse_allreduce(
    const EmbeddingInput& embedding_input, const embedding::EmbeddingOutput& top_grad,
    embedding::Wgrad& wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core_->get_global_gpu_count();
  Wgrad local_reduce_buffer{local_reduce_indices_.attr,
                            local_reduce_indices_.unique_keys,
                            local_reduce_indices_.num_unique_keys,
                            local_reduce_indices_.table_ids,
                            local_reduce_indices_.ev_start_indices,
                            local_reduce_indices_.table_range,
                            wgrad.data};
  local_reduce_index_calculation_.sparse_allreduce_index_calculation.cal_for_sparse_input(
      embedding_input, reduction_indices_, local_reduce_buffer, wgrad, batch_size / num_gpus);

  EmbeddingOutput top_grad_after_average_combiner = top_grad;
  if (std::find(meta_.h_local_combiner_list_.begin(), meta_.h_local_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) != meta_.h_local_combiner_list_.end()) {
    if (top_grad_after_average_combiner.attr.layout == EmbeddingLayout::FeatureMajor) {
      average_combiner_.compute_feature_major(
          embedding_input.fullbatch_bucket_range, top_grad.data, meta_.d_local_lookup_id_list_,
          meta_.d_combiner_list_, meta_.d_ev_size_offset_, batch_size, meta_.max_ev_size_);
    } else {
      average_combiner_.compute_batch_major(embedding_input.fullbatch_bucket_range, top_grad.data,
                                            meta_.d_local_lookup_id_list_, meta_.d_combiner_list_,
                                            meta_.d_ev_size_offset_, batch_size, meta_.max_ev_size_,
                                            meta_.num_lookup_);
    }
    top_grad_after_average_combiner.data = average_combiner_.float_emb_vec_;
    top_grad_after_average_combiner.attr.type = core23::ScalarType::Float;
  }
  local_reduce_.local_reduce(reduction_indices_, top_grad_after_average_combiner,
                             local_reduce_buffer, meta_.d_local_lookup_id_list_,
                             meta_.num_local_lookup_, meta_.num_lookup_, batch_size);

  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  size_t h_num_unique_keys = 0ul;
  HCTR_LIB_THROW(cudaMemcpyAsync(&h_num_unique_keys, wgrad.num_unique_keys.data<uint64_t>(),
                                 sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  uint32_t num_ev_elements;
  HCTR_LIB_THROW(cudaMemcpyAsync(&num_ev_elements,
                                 wgrad.ev_start_indices.data<uint32_t>() + h_num_unique_keys,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  allreduce_comm_.communicate(wgrad.data, num_ev_elements);
}

void UniformDPEmbedding::backward_per_gpu(const EmbeddingInput& embedding_input,
                                          const EmbeddingOutput& top_grad, Wgrad& wgrad,
                                          int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  if (meta_.allreduce_strategy_ == AllreduceStrategy::Dense) {
    backward_per_gpu_with_dense_allreduce(embedding_input, top_grad, wgrad, batch_size);
  } else {
    backward_per_gpu_with_sparse_allreduce(embedding_input, top_grad, wgrad, batch_size);
  }
}

}  // namespace embedding
