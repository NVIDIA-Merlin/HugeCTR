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
#include "HugeCTR/embedding/localized_embedding.hpp"

#include "HugeCTR/include/utils.hpp"
namespace embedding {

UniformLocalizedEmbeddingForward::UniformLocalizedEmbeddingForward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    const GlobalEmbeddingData &global_embedding_data, const EmbeddingShardingParam &sharding_param)
    : core_(core),
      global_embedding_data_(global_embedding_data),
      local_embedding_data_(core, params, sharding_param) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core->get_global_gpu_count();
  auto key_type = params.key_type;
  auto emb_type = params.emb_type;
  model_index_calculation_ = ModelIndexCalculation(
      core, local_embedding_data_.num_local_embedding_, local_embedding_data_.h_local_hotness_list_,
      global_embedding_data_.h_hotness_list_, params.universal_batch_size, key_type);
  model_backward_index_calculation_ = ModelBackwardIndexCalculation(
      core, num_gpus, local_embedding_data_.num_local_embedding_,
      local_embedding_data_.h_local_hotness_list_, local_embedding_data_.h_local_id_space_list_,
      local_embedding_data_.h_local_ev_size_list_, params.universal_batch_size, key_type);
  compress_offset_ = CompressOffset(core, local_embedding_data_.num_local_embedding_ + 1);
  model_forward_ = ModelForward(core, num_gpus, local_embedding_data_.h_local_embedding_list_);
  all2all_comm_ = NcclAll2AllComm(core);
  network_forward_ = NetworkForward(core, num_gpus);
  ragged_network_index_ =
      RaggedNetworkIndex(core, params.universal_batch_size, sharding_param.global_embedding_list,
                         global_embedding_data_.h_ev_size_list_);
  ragged_network_buffer_ =
      RaggedNetworkBuffer(core, params.universal_batch_size, sharding_param.global_embedding_list,
                          global_embedding_data_.h_ev_size_list_, params.emb_type),
  embedding_vec_ = TensorList(
      core_.get(), params.universal_batch_size * local_embedding_data_.num_local_hotness_,
      DeviceType::GPU, TensorScalarType::Float32);

  init_model_comm_buffer(params.universal_batch_size, emb_type);
  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) !=
      local_embedding_data_.h_network_combiner_list_.end()) {
    average_combiner_ =
        AverageCominber(core, num_gpus, local_embedding_data_.h_network_embedding_list_.size(),
                        global_embedding_data_.h_ev_size_list_, params.universal_batch_size);
  }
}

std::vector<size_t> UniformLocalizedEmbeddingForward::get_model_comm_buffer_size(
    int universal_batch_size) {
  int num_gpus = core_->get_global_gpu_count();
  size_t num_ev_elements = 0;
  int batch_size_per_gpu = universal_batch_size / num_gpus;
  for (int embedding_id : local_embedding_data_.h_local_embedding_list_) {
    int ev_size = global_embedding_data_.h_ev_size_list_[embedding_id];
    num_ev_elements += ev_size * batch_size_per_gpu;
  }
  return std::vector<size_t>(num_gpus, num_ev_elements);
}

void UniformLocalizedEmbeddingForward::init_model_comm_buffer(int universal_batch_size,
                                                              DataType emb_type) {
  model_comm_buffer_list_.clear();

  auto model_comm_buffer_size = get_model_comm_buffer_size(universal_batch_size);

  auto buffer_ptr = GetBuffer(core_);
  for (size_t i = 0; i < model_comm_buffer_size.size(); ++i) {
    model_comm_buffer_list_.push_back(
        buffer_ptr->reserve(model_comm_buffer_size[i], DeviceType::GPU, emb_type));
  }
  buffer_ptr->allocate();

  model_comm_buffer_ = TensorList(core_.get(), model_comm_buffer_list_, DeviceType::GPU, emb_type);
}

void UniformLocalizedEmbeddingForward::forward_per_gpu(
    const Tensor &keys, const Tensor &bucket_range, size_t num_keys, const Tensor &sparse_weight,
    ILookup *embedding_table, Tensor &output_buffer, ContextContainer *context_container) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int batch_size = (bucket_range.get_num_elements() - 1) / global_embedding_data_.num_embedding_;

  Tensor model_key, model_offsets;
  size_t num_model_key;
  model_index_calculation_.compute(
      keys, bucket_range, num_keys, local_embedding_data_.d_local_embedding_list_,
      local_embedding_data_.shard_id_, local_embedding_data_.shards_count_, batch_size, &model_key,
      &model_offsets, &num_model_key);
  Tensor id_space_offset;
  compress_offset_.compute(model_offsets, batch_size, &id_space_offset);

  Tensor unique_key, unique_dst_idx, sorted_bucket_id_list, sorted_bucket_id_offset,
      unique_id_space_list, unique_id_space_offset, coordinate_key, coordinate_wgrad_dst_idx;
  size_t num_unique_key;
  model_backward_index_calculation_.compute(
      model_key, num_model_key, model_offsets, id_space_offset,
      local_embedding_data_.d_local_id_space_list_, batch_size, &unique_key, &num_unique_key,
      &unique_dst_idx, &sorted_bucket_id_list, &sorted_bucket_id_offset, &unique_id_space_list,
      &unique_id_space_offset, &coordinate_key, &coordinate_wgrad_dst_idx);

  embedding_table->lookup(model_key, num_model_key, id_space_offset,
                          local_embedding_data_.num_local_embedding_ + 1,
                          local_embedding_data_.d_local_id_space_list_, embedding_vec_);

  model_forward_.compute(embedding_vec_, model_offsets, model_comm_buffer_,
                         local_embedding_data_.d_local_ev_size_list_,
                         local_embedding_data_.d_local_ev_size_offset_, batch_size,
                         local_embedding_data_.max_ev_size_);

  auto model_comm_buffer_size = get_model_comm_buffer_size(batch_size);
  all2all_comm_.communicate(model_comm_buffer_list_, model_comm_buffer_size,
                            ragged_network_buffer_.network_comm_buffer_list_,
                            ragged_network_buffer_.network_comm_buffer_size_);
  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) ==
      local_embedding_data_.h_network_combiner_list_.end()) {
    network_forward_.compute(
        ragged_network_buffer_.network_comm_buffer_, ragged_network_index_.network_ids_,
        ragged_network_index_.network_gpu_ids_, ragged_network_index_.network_offsets_,
        ragged_network_index_.network_dst_lookup_ids_, ragged_network_index_.network_ev_sizes_,
        ragged_network_index_.network_ev_offsets_, output_buffer,
        global_embedding_data_.d_ev_size_offset_, batch_size, global_embedding_data_.max_ev_size_);
  } else {
    network_forward_.compute(
        ragged_network_buffer_.network_comm_buffer_, ragged_network_index_.network_ids_,
        ragged_network_index_.network_gpu_ids_, ragged_network_index_.network_offsets_,
        ragged_network_index_.network_dst_lookup_ids_, ragged_network_index_.network_ev_sizes_,
        ragged_network_index_.network_ev_offsets_, average_combiner_.float_emb_vec_,
        global_embedding_data_.d_ev_size_offset_, batch_size, global_embedding_data_.max_ev_size_);
    average_combiner_.forward(
        bucket_range, output_buffer, local_embedding_data_.d_network_embedding_list_,
        global_embedding_data_.d_combiner_list_, global_embedding_data_.d_ev_size_offset_,
        batch_size, global_embedding_data_.max_ev_size_);
  }

  // for utest
  context_container->pack("model_key", model_key);
  context_container->pack("num_model_key", num_model_key);
  context_container->pack("model_offsets", model_offsets);

  // for backward
  context_container->pack("bucket_range", bucket_range);
  context_container->pack("batch_size", batch_size);
  context_container->pack("unique_key", unique_key);
  context_container->pack("num_unique_key", num_unique_key);
  context_container->pack("unique_dst_idx", unique_dst_idx);
  context_container->pack("sorted_bucket_id_list", sorted_bucket_id_list);
  context_container->pack("sorted_bucket_id_offset", sorted_bucket_id_offset);
  context_container->pack("unique_id_space_list", unique_id_space_list);
  context_container->pack("unique_id_space_offset", unique_id_space_offset);
  context_container->pack("coordinate_key", coordinate_key);
  context_container->pack("coordinate_wgrad_dst_idx", coordinate_wgrad_dst_idx);
  context_container->pack("model_comm_buffer", model_comm_buffer_);
  context_container->pack("model_comm_buffer_size", model_comm_buffer_size);
  context_container->pack("model_comm_buffer_list", model_comm_buffer_list_);
  context_container->pack("ragged_network_index", ragged_network_index_);
  context_container->pack("ragged_network_buffer", ragged_network_buffer_);
}

UniformLocalizedEmbeddingBackward::UniformLocalizedEmbeddingBackward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    const GlobalEmbeddingData &global_embedding_data, const EmbeddingShardingParam &sharding_param)
    : core_(core),
      global_embedding_data_(global_embedding_data),
      local_embedding_data_(core, params, sharding_param) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  int num_gpus = core->get_global_gpu_count();

  network_backward_ = NetworkBackward(core, num_gpus);
  all2all_comm_ = NcclAll2AllComm(core);
  model_backward_ =
      ModelBackward(core, num_gpus, local_embedding_data_.num_local_embedding_,
                    local_embedding_data_.h_local_hotness_list_,
                    local_embedding_data_.h_local_ev_size_list_, params.universal_batch_size,
                    local_embedding_data_.max_ev_size_, global_embedding_data_.num_sms_);
  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) !=
      local_embedding_data_.h_network_combiner_list_.end()) {
    average_combiner_ =
        AverageCominber(core, num_gpus, local_embedding_data_.h_network_embedding_list_.size(),
                        global_embedding_data_.h_ev_size_list_, params.universal_batch_size);
  }
}

void UniformLocalizedEmbeddingBackward::backward_per_gpu(ContextContainer *context_container,
                                                         const Tensor &top_grad, bool do_allreduce,
                                                         Tensor *unique_key, size_t *num_unique_key,
                                                         Tensor *unique_id_space_offset,
                                                         size_t *num_unique_key_id_space_offset,
                                                         Tensor *grad_ev, Tensor *unique_dst_idx) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  auto bucket_range = context_container->unpack<Tensor>("bucket_range");
  auto model_comm_buffer_list =
      context_container->unpack<std::vector<Tensor>>("model_comm_buffer_list");
  auto model_comm_buffer = context_container->unpack<TensorList>("model_comm_buffer");
  auto model_comm_buffer_size =
      context_container->unpack<std::vector<size_t>>("model_comm_buffer_size");
  auto ragged_network_index = context_container->unpack<RaggedNetworkIndex>("ragged_network_index");
  auto ragged_network_buffer =
      context_container->unpack<RaggedNetworkBuffer>("ragged_network_buffer");

  auto batch_size = context_container->unpack<int>("batch_size");
  size_t num_model_key = context_container->unpack<size_t>("num_model_key");

  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) !=
      local_embedding_data_.h_network_combiner_list_.end()) {
    average_combiner_.backward(
        bucket_range, top_grad, local_embedding_data_.d_network_embedding_list_,
        global_embedding_data_.d_combiner_list_, global_embedding_data_.d_ev_size_offset_,
        batch_size, global_embedding_data_.max_ev_size_);
    network_backward_.compute(
        average_combiner_.float_emb_vec_, ragged_network_index.network_ids_,
        ragged_network_index.network_gpu_ids_, ragged_network_index.network_offsets_,
        ragged_network_index.network_dst_lookup_ids_, ragged_network_index.network_ev_sizes_,
        ragged_network_index.network_ev_offsets_, ragged_network_buffer.network_comm_buffer_,
        global_embedding_data_.d_ev_size_offset_, batch_size, global_embedding_data_.max_ev_size_);
  } else {
    network_backward_.compute(
        top_grad, ragged_network_index.network_ids_, ragged_network_index.network_gpu_ids_,
        ragged_network_index.network_offsets_, ragged_network_index.network_dst_lookup_ids_,
        ragged_network_index.network_ev_sizes_, ragged_network_index.network_ev_offsets_,
        ragged_network_buffer.network_comm_buffer_, global_embedding_data_.d_ev_size_offset_,
        batch_size, global_embedding_data_.max_ev_size_);
  }

  all2all_comm_.communicate(ragged_network_buffer.network_comm_buffer_list_,
                            ragged_network_buffer.network_comm_buffer_size_, model_comm_buffer_list,
                            model_comm_buffer_size);
  *unique_key = context_container->unpack<Tensor>("unique_key");
  *num_unique_key = context_container->unpack<size_t>("num_unique_key");
  *unique_id_space_offset = context_container->unpack<Tensor>("unique_id_space_offset");
  *num_unique_key_id_space_offset = unique_id_space_offset->get_num_elements();
  auto sorted_bucket_id_list = context_container->unpack<Tensor>("sorted_bucket_id_list");
  auto sorted_bucket_id_offset = context_container->unpack<Tensor>("sorted_bucket_id_offset");
  *unique_dst_idx = context_container->unpack<Tensor>("unique_dst_idx");
  Tensor coordinate_key = context_container->unpack<Tensor>("coordinate_key");
  Tensor coordinate_wgrad_dst_idx = context_container->unpack<Tensor>("coordinate_wgrad_dst_idx");

  model_backward_.compute(model_comm_buffer, *unique_dst_idx, sorted_bucket_id_list,
                          sorted_bucket_id_offset, *num_unique_key, coordinate_key,
                          coordinate_wgrad_dst_idx, local_embedding_data_.d_local_ev_size_offset_,
                          batch_size, local_embedding_data_.max_ev_size_, num_model_key, grad_ev);
}
}  // namespace embedding
