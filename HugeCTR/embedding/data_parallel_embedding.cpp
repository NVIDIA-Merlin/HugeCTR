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
#include "data_parallel_embedding.hpp"

#include "HugeCTR/include/utils.hpp"
namespace embedding {

UniformDataParallelEmbeddingMeta::UniformDataParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam& ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup), h_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto& lookup_params = ebc_param.lookup_params;
  auto buffer_ptr = GetBuffer(core);
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
  d_ev_size_offset_ =
      buffer_ptr->reserve({h_ev_size_offset_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_ev_size_offset_.copy_from(h_ev_size_offset_);

  d_combiner_list_ =
      buffer_ptr->reserve({h_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  buffer_ptr->allocate();
  d_combiner_list_.copy_from(h_combiner_list_);

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());

  d_local_lookup_id_list_ = buffer_ptr->reserve({h_local_lookup_id_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_lookup_id_list_.copy_from(h_local_lookup_id_list_);

  d_local_ev_size_list_ =
      buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_ev_size_list_.copy_from(h_local_ev_size_list_);

  d_local_table_id_list_ = buffer_ptr->reserve({h_local_table_id_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_table_id_list_.copy_from(h_local_table_id_list_);
  wgrad_attr.init(core, ebc_param, grouped_id);

  if (ebc_param.indices_only_ && !ebc_param.table_id_to_vocabulary_size.empty()) {
    h_table_id_to_global_start_indices = ebc_param.get_table_id_to_global_start_indices();
    table_id_to_global_start_indices = buffer_ptr->reserve(
        {h_table_id_to_global_start_indices.size()}, DeviceType::GPU, TensorScalarType::Int32);
    buffer_ptr->allocate();
    table_id_to_global_start_indices.copy_from(h_table_id_to_global_start_indices);

    std::vector<int> h_table_id_to_allreduce_buffer_start_indices(ebc_param.num_table, 0);
    int cnt = 0;
    for (int i = 0; i < wgrad_attr.num_table; ++i) {
      int table_id = wgrad_attr.h_sorted_unique_table_ids[i];
      h_table_id_to_allreduce_buffer_start_indices[table_id] = cnt;
      cnt += ebc_param.table_id_to_vocabulary_size[table_id];
    }
    table_id_to_allreduce_buffer_start_indices =
        buffer_ptr->reserve({h_table_id_to_allreduce_buffer_start_indices.size()}, DeviceType::GPU,
                            TensorScalarType::Int32);
    buffer_ptr->allocate();
    table_id_to_allreduce_buffer_start_indices.copy_from(
        h_table_id_to_allreduce_buffer_start_indices);
  }
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

  embedding_vec_ = TensorList(core_.get(), universal_batch_size * meta_.num_local_hotness_,
                              DeviceType::GPU, TensorScalarType::Float32);

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
  if (params.indices_only_) {
    int vocubulary_size_sum = meta_.h_table_id_to_global_start_indices.back();
    int end_bit = static_cast<int>(std::log2(static_cast<float>(vocubulary_size_sum) + 1));
    IndicesSort indices_sort{core,
                             meta_.table_id_to_global_start_indices,
                             end_bit,
                             meta_.num_local_hotness_,
                             params.universal_batch_size / num_gpus,
                             key_type};
    local_reduce_index_calculation_.dense_allreduce_index_calculation = {
        core, local_reduce_index_calculation, indices_sort, cal_dst_ids, segmented_unique};
  } else {
    SegmentedSortDevice segmented_sort{core, meta_.num_local_hotness_,
                                       params.universal_batch_size / num_gpus,
                                       meta_.wgrad_attr.num_table, key_type};
    SparseAllreduceCalEVStartIndicesStorage sparse_allreduce_storage{
        core, meta_.wgrad_attr.num_table, meta_.num_local_hotness_,
        params.universal_batch_size / num_gpus, key_type};
    local_reduce_index_calculation_.sparse_allreduce_index_calculation = {
        core,
        local_reduce_index_calculation,
        segmented_sort,
        cal_dst_ids,
        segmented_unique,
        sparse_allreduce_storage};
  }
  local_reduce_.init(core, meta_.kernel_params, meta_.max_ev_size_,
                     meta_.num_local_hotness_ * (params.universal_batch_size / num_gpus));
}

void UniformDPEmbedding::forward_per_gpu(const EmbeddingInput& embedding_input,
                                         ILookup* embedding_table,
                                         EmbeddingOutput& embedding_output, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  Tensor num_key_per_lookup_offset;
  compress_offset_.compute(embedding_input.bucket_range, batch_size_per_gpu,
                           &num_key_per_lookup_offset);

  embedding_table->lookup(embedding_input.keys, embedding_input.h_num_keys,
                          num_key_per_lookup_offset, meta_.num_local_lookup_ + 1,
                          meta_.d_local_table_id_list_, embedding_vec_);
  dp_model_forward_.compute(embedding_vec_, embedding_input.bucket_range,
                            meta_.d_local_lookup_id_list_, embedding_output, batch_size_per_gpu);
}

void UniformDPEmbedding::backward_per_gpu_for_indices_only(
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
      embedding_input, meta_.table_id_to_allreduce_buffer_start_indices, wgrad.ev_start_indices,
      reduction_indices_, local_reduce_buffer, batch_size / num_gpus);

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
    top_grad_after_average_combiner.attr.type = core::TensorScalarType::Float32;
  }
  local_reduce_.local_reduce(reduction_indices_, top_grad_after_average_combiner,
                             local_reduce_buffer, meta_.d_local_lookup_id_list_,
                             meta_.num_local_lookup_, meta_.num_lookup_, batch_size);

  allreduce_comm_.communicate(wgrad.data, wgrad.data.get_num_elements());
}

void UniformDPEmbedding::backward_per_gpu_for_dynamic_table(
    const EmbeddingInput& embedding_input, const embedding::EmbeddingOutput& top_grad,
    embedding::Wgrad& wgrad, int batch_size) {
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
    top_grad_after_average_combiner.attr.type = core::TensorScalarType::Float32;
  }
  local_reduce_.local_reduce(reduction_indices_, top_grad_after_average_combiner,
                             local_reduce_buffer, meta_.d_local_lookup_id_list_,
                             meta_.num_local_lookup_, meta_.num_lookup_, batch_size);

  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  size_t h_num_unique_keys = 0ul;
  HCTR_LIB_THROW(cudaMemcpyAsync(&h_num_unique_keys, wgrad.num_unique_keys.get<size_t>(),
                                 sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  uint32_t num_ev_elements;
  HCTR_LIB_THROW(cudaMemcpyAsync(&num_ev_elements,
                                 wgrad.ev_start_indices.get<uint32_t>() + h_num_unique_keys,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  allreduce_comm_.communicate(wgrad.data, num_ev_elements);
}

void UniformDPEmbedding::backward_per_gpu(const EmbeddingInput& embedding_input,
                                          const EmbeddingOutput& top_grad, Wgrad& wgrad,
                                          int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  if (!meta_.table_id_to_allreduce_buffer_start_indices.empty()) {
    backward_per_gpu_for_indices_only(embedding_input, top_grad, wgrad, batch_size);
  } else {
    backward_per_gpu_for_dynamic_table(embedding_input, top_grad, wgrad, batch_size);
  }
}
}  // namespace embedding
