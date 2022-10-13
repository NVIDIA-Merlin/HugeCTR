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
#include "HugeCTR/embedding/model_parallel_embedding.hpp"

#include "HugeCTR/include/utils.hpp"
namespace embedding {

UniformModelParallelEmbedding::UniformModelParallelEmbedding(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    size_t grouped_id)
    : core_(core), meta_(core, params, grouped_id) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core->get_global_gpu_count();
  auto key_type = params.key_type;
  auto emb_type = params.emb_type;
  model_index_calculation_ =
      ModelIndexCalculation(core, meta_.num_local_lookup_, meta_.num_local_hotness_,
                            meta_.hotness_sum_, params.universal_batch_size, key_type);
  model_backward_index_calculation_ = ModelBackwardIndexCalculation(
      core, num_gpus, meta_.num_local_lookup_, meta_.h_local_hotness_list_,
      meta_.h_local_table_id_list_, meta_.h_local_ev_size_list_, params.universal_batch_size,
      key_type);
  compress_offset_ = CompressOffset(core, meta_.num_local_lookup_ + 1);
  model_forward_ = ModelForward(core, num_gpus, meta_.h_local_lookup_id_list_);
  all2all_comm_ = NcclAll2AllComm(core);
  network_forward_ = NetworkForward(core, num_gpus);
  network_backward_ = NetworkBackward(core, num_gpus);
  model_backward_ = ModelBackward(core, num_gpus, meta_.num_local_lookup_,
                                  meta_.h_local_hotness_list_, meta_.h_local_ev_size_list_,
                                  params.universal_batch_size, meta_.max_ev_size_, meta_.num_sms_);

  embedding_vec_ = TensorList(core_.get(), params.universal_batch_size * meta_.num_local_hotness_,
                              DeviceType::GPU, TensorScalarType::Float32);

  init_model_comm_buffer(params.universal_batch_size, emb_type);
  init_network_comm_buffer(params.universal_batch_size, emb_type);
}

std::vector<size_t> UniformModelParallelEmbedding::get_model_comm_buffer_size(
    int universal_batch_size) {
  int num_gpus = core_->get_global_gpu_count();
  size_t num_ev_elements = 0;
  int batch_size_per_gpu = universal_batch_size / num_gpus;
  for (int lookup_id : meta_.h_local_lookup_id_list_) {
    int ev_size = meta_.h_ev_size_list_[lookup_id];
    num_ev_elements += ev_size * batch_size_per_gpu;
  }
  return std::vector<size_t>(num_gpus, num_ev_elements);
}

void UniformModelParallelEmbedding::init_model_comm_buffer(int universal_batch_size,
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

std::vector<size_t> UniformModelParallelEmbedding::get_network_comm_buffer_size(
    int universal_batch_size) {
  int num_gpus = core_->get_global_gpu_count();
  int batch_size_per_gpu = universal_batch_size / num_gpus;

  std::vector<size_t> network_comm_buffer_size;
  for (int global_gpu_id = 0; global_gpu_id < num_gpus; ++global_gpu_id) {
    auto &remote_embedding_list = meta_.h_global_lookup_id_list_[global_gpu_id];
    size_t num_ev_elements = 0;
    for (int embedding_id : remote_embedding_list) {
      num_ev_elements += meta_.h_ev_size_list_[embedding_id] * batch_size_per_gpu;
    }
    network_comm_buffer_size.push_back(num_ev_elements);
  }
  return network_comm_buffer_size;
}

void UniformModelParallelEmbedding::init_network_comm_buffer(int universal_batch_size,
                                                             DataType emb_type) {
  network_comm_buffer_list_.clear();

  auto network_comm_buffer_size = get_network_comm_buffer_size(universal_batch_size);

  auto buffer_ptr = GetBuffer(core_);
  for (size_t i = 0; i < network_comm_buffer_size.size(); ++i) {
    network_comm_buffer_list_.push_back(
        buffer_ptr->reserve(network_comm_buffer_size[i], DeviceType::GPU, emb_type));
  }
  buffer_ptr->allocate();

  network_comm_buffer_ =
      TensorList(core_.get(), network_comm_buffer_list_, DeviceType::GPU, emb_type);
}

void UniformModelParallelEmbedding::forward_per_gpu(const Tensor &keys, const Tensor &bucket_range,
                                                    size_t num_keys, ILookup *embedding_table,
                                                    Tensor &output_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  batch_size_ = batch_size;
  bucket_range_ = bucket_range;

  Tensor model_key, model_offsets;
  size_t num_model_key;
  model_index_calculation_.compute(keys, bucket_range, num_keys, meta_.d_local_lookup_id_list_,
                                   meta_.d_local_shard_id_list_, meta_.d_local_num_shards_list_,
                                   batch_size, &model_key, &model_offsets, &num_model_key);
  Tensor num_key_per_lookup_offset;
  compress_offset_.compute(model_offsets, batch_size, &num_key_per_lookup_offset);

  model_key_ = model_key;
  model_offsets_ = model_offsets;
  num_model_key_ = num_model_key;
  num_key_per_lookup_offset_ = num_key_per_lookup_offset;

  embedding_table->lookup(model_key, num_model_key, num_key_per_lookup_offset,
                          meta_.num_local_lookup_ + 1, meta_.d_local_table_id_list_,
                          embedding_vec_);

  model_forward_.compute(embedding_vec_, model_offsets, model_comm_buffer_,
                         meta_.d_local_ev_size_list_, meta_.d_local_ev_size_offset_, batch_size,
                         meta_.max_ev_size_);

  auto model_comm_buffer_size = get_model_comm_buffer_size(batch_size);
  auto network_comm_buffer_size = get_network_comm_buffer_size(batch_size);
  all2all_comm_.communicate(model_comm_buffer_list_, model_comm_buffer_size,
                            network_comm_buffer_list_, network_comm_buffer_size);
  network_forward_.compute(bucket_range, meta_.d_combiner_list_, network_comm_buffer_,
                           meta_.network_ids_, meta_.network_gpu_ids_, meta_.network_offsets_,
                           meta_.network_dst_lookup_ids_, meta_.network_ev_sizes_,
                           meta_.network_ev_offsets_, output_buffer, meta_.d_ev_size_offset_,
                           batch_size, meta_.max_ev_size_);
}

void UniformModelParallelEmbedding::backward_per_gpu(const Tensor &top_grad, bool do_allreduce,
                                                     Tensor *unique_key, size_t *num_unique_key,
                                                     Tensor *num_unique_key_per_table_offset,
                                                     size_t *num_table_offset,
                                                     Tensor *table_id_list, Tensor *wgrad,
                                                     Tensor *wgrad_idx_offset) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  Tensor sorted_bucket_id_list, sorted_bucket_id_offset, coordinate_key, coordinate_wgrad_dst_idx;
  model_backward_index_calculation_.compute(
      model_key_, num_model_key_, model_offsets_, num_key_per_lookup_offset_,
      meta_.d_local_table_id_list_, batch_size_, unique_key, num_unique_key, wgrad_idx_offset,
      &sorted_bucket_id_list, &sorted_bucket_id_offset, table_id_list,
      num_unique_key_per_table_offset, &coordinate_key, &coordinate_wgrad_dst_idx);
  *num_table_offset = num_unique_key_per_table_offset->get_num_elements();

  network_backward_.compute(bucket_range_, meta_.d_combiner_list_, top_grad, meta_.network_ids_,
                            meta_.network_gpu_ids_, meta_.network_offsets_,
                            meta_.network_dst_lookup_ids_, meta_.network_ev_sizes_,
                            meta_.network_ev_offsets_, network_comm_buffer_,
                            meta_.d_ev_size_offset_, batch_size_, meta_.max_ev_size_);

  auto model_comm_buffer_size = get_model_comm_buffer_size(batch_size_);
  auto network_comm_buffer_size = get_network_comm_buffer_size(batch_size_);
  all2all_comm_.communicate(network_comm_buffer_list_, network_comm_buffer_size,
                            model_comm_buffer_list_, model_comm_buffer_size);

  model_backward_.compute(model_comm_buffer_, *wgrad_idx_offset, sorted_bucket_id_list,
                          sorted_bucket_id_offset, *num_unique_key, coordinate_key,
                          coordinate_wgrad_dst_idx, meta_.d_local_ev_size_offset_, batch_size_,
                          meta_.max_ev_size_, num_model_key_, wgrad);
}

}  // namespace embedding
