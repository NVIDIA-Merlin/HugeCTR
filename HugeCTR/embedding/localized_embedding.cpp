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

namespace embedding {

RaggedNetworkBuffer::RaggedNetworkBuffer(std::shared_ptr<CoreResourceManager> core, int batch_size,
                                         const std::vector<std::vector<int>> &global_embedding_list,
                                         const std::vector<int> &ev_size_list, DataType emb_type) {
  CudaDeviceContext context(core->get_device_id());
  int num_gpus = core->get_global_gpu_count();
  int batch_size_per_gpu = batch_size / num_gpus;

  auto init_network_comm_buffer = [&] {
    for (int global_gpu_id = 0; global_gpu_id < num_gpus; ++global_gpu_id) {
      auto &remote_embedding_list = global_embedding_list[global_gpu_id];
      size_t num_ev_elements = 0;
      for (int embedding_id : remote_embedding_list) {
        num_ev_elements += ev_size_list[embedding_id] * batch_size_per_gpu;
      }
      network_comm_buffer_size_.push_back(num_ev_elements);
    }
    auto buffer_ptr = GetBuffer(core);
    for (size_t i = 0; i < network_comm_buffer_size_.size(); ++i) {
      network_comm_buffer_list_.push_back(
          buffer_ptr->reserve(network_comm_buffer_size_[i], DeviceType::GPU, emb_type));
    }
    buffer_ptr->allocate();

    network_comm_buffer_ =
        TensorList(core.get(), network_comm_buffer_list_, DeviceType::GPU, emb_type);
  };

  auto init_netwok_idx = [&] {
    std::vector<int> network_idx_list;
    std::vector<int> network_offset_list{0};
    std::vector<int> network_dst_list;

    std::vector<int> dst_embedding_id_list;
    for (auto &vec : global_embedding_list) {
      dst_embedding_id_list.insert(dst_embedding_id_list.end(), vec.begin(), vec.end());
    }

    std::sort(dst_embedding_id_list.begin(), dst_embedding_id_list.end());
    auto last = std::unique(dst_embedding_id_list.begin(), dst_embedding_id_list.end());
    dst_embedding_id_list.erase(last, dst_embedding_id_list.end());
    for (int dst_embedding_id : dst_embedding_id_list) {
      for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
        network_dst_list.push_back(batch_size_per_gpu * dst_embedding_id + batch_id);
      }
    }

    std::vector<int> num_embedding_offset{0};
    for (auto &vec : global_embedding_list) {
      num_embedding_offset.push_back(vec.size());
    }
    std::partial_sum(num_embedding_offset.begin(), num_embedding_offset.end(),
                     num_embedding_offset.begin());

    std::vector<int> network_embedding_list;
    std::vector<int> network_embedding_offset{0};

    network_embedding_offset.assign(dst_embedding_id_list.size() + 1, 0);

    for (int local_embedding_id = 0;
         local_embedding_id < static_cast<int>(dst_embedding_id_list.size());
         ++local_embedding_id) {
      int dst_embedding_id = dst_embedding_id_list[local_embedding_id];

      for (int src_gpu_id = 0; src_gpu_id < num_gpus; ++src_gpu_id) {
        auto iter = std::find(global_embedding_list[src_gpu_id].begin(),
                              global_embedding_list[src_gpu_id].end(), dst_embedding_id);
        if (iter == global_embedding_list[src_gpu_id].end()) continue;
        int idx = std::distance(global_embedding_list[src_gpu_id].begin(), iter);

        network_embedding_list.push_back(num_embedding_offset[src_gpu_id] + idx);
        network_embedding_offset[1 + local_embedding_id] += 1;
      }
    }
    std::inclusive_scan(network_embedding_offset.begin(), network_embedding_offset.end(),
                        network_embedding_offset.begin());

    for (size_t i = 0; i < dst_embedding_id_list.size(); ++i) {
      int start = network_embedding_offset[i];
      int end = network_embedding_offset[i + 1];

      for (int batch_id = 0; batch_id < batch_size_per_gpu; ++batch_id) {
        for (int r = start; r < end; ++r) {
          int embedding_id = network_embedding_list[r];
          network_idx_list.push_back(embedding_id * batch_size_per_gpu + batch_id);
        }
        network_offset_list.push_back(end - start);
      }
    }

    std::inclusive_scan(network_offset_list.begin(), network_offset_list.end(),
                        network_offset_list.begin());

    auto buffer_ptr = GetBuffer(core);
    core::DataType data_type = {HugeCTR::TensorScalarType::Int32};
    network_idx_ = buffer_ptr->reserve({network_idx_list.size()}, DeviceType::GPU, data_type);
    network_offset_ = buffer_ptr->reserve({network_offset_list.size()}, DeviceType::GPU, data_type);
    network_dst_ = buffer_ptr->reserve({network_dst_list.size()}, DeviceType::GPU, data_type);
    buffer_ptr->allocate();

    network_idx_.copy_from(network_idx_list);
    network_offset_.copy_from(network_offset_list);
    network_dst_.copy_from(network_dst_list);
  };

  auto init_network_view_idx = [&] {
    std::vector<int> gpu_idx_offset{0};
    std::vector<std::vector<int>> global_ev_offset;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      auto &local_embedding_list = global_embedding_list[gpu_id];
      int num_local_embedding = local_embedding_list.size();
      gpu_idx_offset.push_back(num_local_embedding * batch_size_per_gpu);

      std::vector<int> local_ev_offset{0};
      for (int embedding_id : local_embedding_list) {
        local_ev_offset.push_back(ev_size_list[embedding_id]);
      }
      global_ev_offset.push_back(local_ev_offset);
    }

    std::partial_sum(gpu_idx_offset.begin(), gpu_idx_offset.end(), gpu_idx_offset.begin());
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      std::partial_sum(global_ev_offset[gpu_id].begin(), global_ev_offset[gpu_id].end(),
                       global_ev_offset[gpu_id].begin());
    }

    auto buffer_ptr = GetBuffer(core);
    gpu_idx_offset_ =
        buffer_ptr->reserve({gpu_idx_offset.size()}, DeviceType::GPU, TensorScalarType::Int32);

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      global_ev_offset_list_.push_back(buffer_ptr->reserve(
          global_ev_offset[gpu_id].size(), DeviceType::GPU, TensorScalarType::Int32));
    }
    buffer_ptr->allocate();

    global_ev_offset_ =
        TensorList(core.get(), global_ev_offset_list_, DeviceType::GPU, TensorScalarType::Int32);

    gpu_idx_offset_.copy_from(gpu_idx_offset);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      global_ev_offset_list_[gpu_id].copy_from(global_ev_offset[gpu_id]);
    }
  };
  init_network_comm_buffer();
  init_netwok_idx();
  init_network_view_idx();
}

UniformLocalizedEmbeddingForward::UniformLocalizedEmbeddingForward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    const GlobalEmbeddingData &global_embedding_data, const EmbeddingShardingParam &sharding_param)
    : core_(core),
      global_embedding_data_(global_embedding_data),
      local_embedding_data_(core, params, sharding_param) {
  CudaDeviceContext context(core_->get_device_id());
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
  ragged_network_buffer_ =
      RaggedNetworkBuffer(core, params.universal_batch_size, sharding_param.global_embedding_list,
                          global_embedding_data_.h_ev_size_list_, params.emb_type),
  embedding_vec_ = TensorList(
      core_.get(), params.universal_batch_size * local_embedding_data_.num_local_hotness_,
      DeviceType::GPU, TensorScalarType::Float32);

  init_model_comm_buffer(params.universal_batch_size, emb_type);
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
  CudaDeviceContext context(core_->get_device_id());
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
      unique_id_space_list, unique_id_space_offset;
  size_t num_unique_key;
  model_backward_index_calculation_.compute(
      model_key, num_model_key, model_offsets, id_space_offset,
      local_embedding_data_.d_local_id_space_list_, batch_size, &unique_key, &num_unique_key,
      &unique_dst_idx, &sorted_bucket_id_list, &sorted_bucket_id_offset, &unique_id_space_list,
      &unique_id_space_offset);

  embedding_table->lookup(model_key, num_model_key, id_space_offset,
                          local_embedding_data_.num_local_embedding_,
                          local_embedding_data_.d_local_id_space_list_, embedding_vec_);

  model_forward_.compute(embedding_vec_, model_offsets, model_comm_buffer_,
                         local_embedding_data_.d_local_ev_size_list_,
                         local_embedding_data_.d_local_ev_size_offset_, batch_size);

  auto model_comm_buffer_size = get_model_comm_buffer_size(batch_size);
  all2all_comm_.communicate(model_comm_buffer_list_, model_comm_buffer_size,
                            ragged_network_buffer_.network_comm_buffer_list_,
                            ragged_network_buffer_.network_comm_buffer_size_);

  network_forward_.compute(
      bucket_range, global_embedding_data_.d_combiner_list_,
      ragged_network_buffer_.network_comm_buffer_, ragged_network_buffer_.gpu_idx_offset_,
      ragged_network_buffer_.global_ev_offset_, ragged_network_buffer_.network_idx_,
      ragged_network_buffer_.network_offset_, ragged_network_buffer_.network_dst_, output_buffer,
      global_embedding_data_.d_ev_size_offset_, batch_size);

  // for utest
  context_container->pack("model_key", model_key);
  context_container->pack("num_model_key", num_model_key);
  context_container->pack("model_offsets", model_offsets);

  // for backward
  context_container->pack("batch_size", batch_size);
  context_container->pack("unique_key", unique_key);
  context_container->pack("num_unique_key", num_unique_key);
  context_container->pack("unique_dst_idx", unique_dst_idx);
  context_container->pack("sorted_bucket_id_list", sorted_bucket_id_list);
  context_container->pack("sorted_bucket_id_offset", sorted_bucket_id_offset);
  context_container->pack("unique_id_space_list", unique_id_space_list);
  context_container->pack("unique_id_space_offset", unique_id_space_offset);
  context_container->pack("model_comm_buffer", model_comm_buffer_);
  context_container->pack("model_comm_buffer_size", model_comm_buffer_size);
  context_container->pack("model_comm_buffer_list", model_comm_buffer_list_);
  context_container->pack("ragged_network_buffer", ragged_network_buffer_);
}

UniformLocalizedEmbeddingBackward::UniformLocalizedEmbeddingBackward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    const GlobalEmbeddingData &global_embedding_data, const EmbeddingShardingParam &sharding_param)
    : core_(core),
      global_embedding_data_(global_embedding_data),
      local_embedding_data_(core, params, sharding_param) {
  CudaDeviceContext context(core_->get_device_id());

  int num_gpus = core->get_global_gpu_count();

  network_backward_ = NetworkBackward(core, num_gpus);
  all2all_comm_ = NcclAll2AllComm(core);
  model_backward_ =
      ModelBackward(core, num_gpus, local_embedding_data_.num_local_embedding_,
                    local_embedding_data_.h_local_hotness_list_,
                    local_embedding_data_.h_local_ev_size_list_, params.universal_batch_size);
}

void UniformLocalizedEmbeddingBackward::backward_per_gpu(ContextContainer *context_container,
                                                         const Tensor &top_grad, bool do_allreduce,
                                                         Tensor *unique_key, size_t *num_unique_key,
                                                         Tensor *unique_id_space_offset,
                                                         size_t *num_unique_key_id_space_offset,
                                                         Tensor *grad_ev, Tensor *unique_dst_idx) {
  CudaDeviceContext context(core_->get_device_id());

  auto model_comm_buffer_list =
      context_container->unpack<std::vector<Tensor>>("model_comm_buffer_list");
  auto model_comm_buffer = context_container->unpack<TensorList>("model_comm_buffer");
  auto model_comm_buffer_size =
      context_container->unpack<std::vector<size_t>>("model_comm_buffer_size");
  auto ragged_network_buffer =
      context_container->unpack<RaggedNetworkBuffer>("ragged_network_buffer");

  auto batch_size = context_container->unpack<int>("batch_size");

  network_backward_.compute(
      top_grad, global_embedding_data_.d_ev_size_offset_, ragged_network_buffer.gpu_idx_offset_,
      ragged_network_buffer.global_ev_offset_, ragged_network_buffer.network_idx_,
      ragged_network_buffer.network_offset_, ragged_network_buffer.network_dst_,
      ragged_network_buffer.network_comm_buffer_, batch_size);

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

  model_backward_.compute(model_comm_buffer, *unique_dst_idx, sorted_bucket_id_list,
                          sorted_bucket_id_offset, *num_unique_key,
                          local_embedding_data_.d_local_ev_size_offset_, batch_size, grad_ev);
}
}  // namespace embedding
