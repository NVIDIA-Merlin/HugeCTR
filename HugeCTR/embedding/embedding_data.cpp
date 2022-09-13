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
#include "embedding_data.hpp"

#include "HugeCTR/include/utils.hpp"
#include "common.hpp"
namespace embedding {

GlobalEmbeddingData::GlobalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                                         const EmbeddingCollectionParam &params)
    : core_(core), num_embedding_(params.num_embedding), h_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  auto &embedding_params = params.embedding_params;

  for (int embedding_id = 0; embedding_id < num_embedding_; ++embedding_id) {
    h_hotness_list_.push_back(embedding_params[embedding_id].hotness);
    h_ev_size_list_.push_back(embedding_params[embedding_id].ev_size);
    h_combiner_list_.push_back(static_cast<char>(embedding_params[embedding_id].combiner));
  }
  std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                   std::back_inserter(h_ev_size_offset_));

  num_hotness_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);
  max_ev_size_ = h_ev_size_list_.size() > 0
                     ? *std::max_element(h_ev_size_list_.begin(), h_ev_size_list_.end())
                     : 0;

  // cudaDeviceProp device_prop;
  // cudaGetDeviceProperties(&device_prop, 0);
  // num_sms_ = device_prop.multiProcessorCount;
  // FIX: cudaGetDeviceProperties get ,cost too much time, need remove it to the start of program ,
  // not use per iteration,for now fix the num_sms_
  num_sms_ = 108;

  // init device bufffer
  auto buffer_ptr = GetBuffer(core_);
  d_hotness_list_ =
      buffer_ptr->reserve({h_hotness_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_ev_size_list_ =
      buffer_ptr->reserve({h_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_ev_size_offset_ =
      buffer_ptr->reserve({h_ev_size_offset_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_combiner_list_ =
      buffer_ptr->reserve({h_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  buffer_ptr->allocate();

  d_hotness_list_.copy_from(h_hotness_list_);
  d_ev_size_list_.copy_from(h_ev_size_list_);
  d_ev_size_offset_.copy_from(h_ev_size_offset_);
  d_combiner_list_.copy_from(h_combiner_list_);
}

LocalEmbeddingData::LocalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                                       const EmbeddingCollectionParam &params,
                                       const EmbeddingShardingParam &sharding_param)
    : core_(core),
      num_local_embedding_(sharding_param.local_embedding_list.size()),
      shard_id_(sharding_param.shard_id),
      shards_count_(sharding_param.shards_count),
      h_local_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  auto &embedding_params = params.embedding_params;

  h_local_embedding_list_ = sharding_param.local_embedding_list;
  h_global_embedding_list_ = sharding_param.global_embedding_list;

  for (int embedding_id : sharding_param.local_embedding_list) {
    h_local_id_space_list_.push_back(embedding_params[embedding_id].id_space);
    h_local_hotness_list_.push_back(embedding_params[embedding_id].hotness);
    h_local_ev_size_list_.push_back(embedding_params[embedding_id].ev_size);
    h_local_combiner_list_.push_back(static_cast<char>(embedding_params[embedding_id].combiner));
  }
  std::partial_sum(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end(),
                   std::back_inserter(h_local_ev_size_offset_));
  max_ev_size_ = h_local_ev_size_list_.size() > 0
                     ? *std::max_element(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end())
                     : 0;

  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);

  h_network_embedding_list_.clear();
  for (auto &vec : h_global_embedding_list_) {
    h_network_embedding_list_.insert(h_network_embedding_list_.end(), vec.begin(), vec.end());
  }

  std::sort(h_network_embedding_list_.begin(), h_network_embedding_list_.end());
  auto last = std::unique(h_network_embedding_list_.begin(), h_network_embedding_list_.end());
  h_network_embedding_list_.erase(last, h_network_embedding_list_.end());

  for (int embedding_id : h_network_embedding_list_) {
    h_network_combiner_list_.push_back(
        static_cast<char>(params.embedding_params[embedding_id].combiner));
  }
  // init device bufffer
  auto buffer_ptr = GetBuffer(core_);
  d_local_embedding_list_ = buffer_ptr->reserve({h_local_embedding_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  d_local_id_space_list_ = buffer_ptr->reserve({h_local_id_space_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  d_local_hotness_list_ =
      buffer_ptr->reserve({h_local_hotness_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_local_ev_size_list_ =
      buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_local_ev_size_offset_ = buffer_ptr->reserve({h_local_ev_size_offset_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  d_local_combiner_list_ =
      buffer_ptr->reserve({h_local_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  d_network_embedding_list_ = buffer_ptr->reserve({h_network_embedding_list_.size()},
                                                  DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();

  d_local_embedding_list_.copy_from(h_local_embedding_list_);
  d_local_id_space_list_.copy_from(h_local_id_space_list_);
  d_local_hotness_list_.copy_from(h_local_hotness_list_);
  d_local_ev_size_list_.copy_from(h_local_ev_size_list_);
  d_local_ev_size_offset_.copy_from(h_local_ev_size_offset_);
  d_local_combiner_list_.copy_from(h_local_combiner_list_);
  d_network_embedding_list_.copy_from(h_network_embedding_list_);
}

LocalEmbeddingData::LocalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                                       const EmbeddingCollectionParam &params,
                                       const EmbeddingShardParam &shard_param)
    : core_(core), h_local_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  size_t global_gpu_count = core->get_global_gpu_count();
  int global_gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(shard_param.shard_matrix.size() == global_gpu_count,
                  "shard matrix should contain global_gpu_count row.");
  HCTR_CHECK_HINT(
      shard_param.shard_matrix[global_gpu_id].size() == static_cast<size_t>(params.num_embedding),
      "shard matrix should contain num_embedding column.");

  num_local_embedding_ = 0;
  num_local_hotness_ = 0;
  shard_id_ = -1;
  for (int embedding_id = 0; embedding_id < params.num_embedding; ++embedding_id) {
    if (shard_param.shard_matrix[global_gpu_id][embedding_id] < 0) {
      continue;
    }
    h_local_embedding_list_.push_back(embedding_id);
    h_local_id_space_list_.push_back(params.embedding_params[embedding_id].id_space);
    h_local_hotness_list_.push_back(params.embedding_params[embedding_id].hotness);
    h_local_ev_size_list_.push_back(params.embedding_params[embedding_id].ev_size);
    h_local_combiner_list_.push_back(
        static_cast<char>(params.embedding_params[embedding_id].combiner));
    num_local_embedding_ += 1;
    num_local_hotness_ += params.embedding_params[embedding_id].hotness;
    if (shard_id_ < 0) {
      shard_id_ = shard_param.shard_matrix[global_gpu_id][embedding_id];
    } else {
      HCTR_CHECK_HINT(shard_id_ == shard_param.shard_matrix[global_gpu_id][embedding_id],
                      "Current implementation does not support multiple shard id in one gpu");
    }
  }
  std::partial_sum(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end(),
                   std::back_inserter(h_local_ev_size_offset_));
  max_ev_size_ = h_local_ev_size_list_.size() > 0
                     ? *std::max_element(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end())
                     : 0;

  h_global_embedding_list_.resize(global_gpu_count);
  for (size_t gpu_id = 0; gpu_id < global_gpu_count; ++gpu_id) {
    for (int embedding_id = 0; embedding_id < params.num_embedding; ++embedding_id) {
      if (shard_param.shard_matrix[gpu_id][embedding_id] < 0) {
        continue;
      }
      h_global_embedding_list_[gpu_id].push_back(embedding_id);
    }
  }

  h_network_embedding_list_.clear();
  for (auto &vec : h_global_embedding_list_) {
    h_network_embedding_list_.insert(h_network_embedding_list_.end(), vec.begin(), vec.end());
  }

  std::sort(h_network_embedding_list_.begin(), h_network_embedding_list_.end());
  auto last = std::unique(h_network_embedding_list_.begin(), h_network_embedding_list_.end());
  h_network_embedding_list_.erase(last, h_network_embedding_list_.end());

  for (int embedding_id : h_network_embedding_list_) {
    h_network_combiner_list_.push_back(
        static_cast<char>(params.embedding_params[embedding_id].combiner));
  }

  shards_count_ = -1;
  for (int embedding_id : h_local_embedding_list_) {
    int shard_count = 0;
    for (size_t gpu_id = 0; gpu_id < global_gpu_count; ++gpu_id) {
      if (shard_param.shard_matrix[gpu_id][embedding_id] < 0) {
        continue;
      }
      shard_count += 1;
    }
    if (shards_count_ < 0) {
      shards_count_ = shard_count;
    } else {
      HCTR_CHECK_HINT(shards_count_ == shard_count,
                      "Current implementation does not support multiple num_sharding");
    }
  }

  // init device bufffer
  auto buffer_ptr = GetBuffer(core_);
  d_local_embedding_list_ = buffer_ptr->reserve({h_local_embedding_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  d_local_id_space_list_ = buffer_ptr->reserve({h_local_id_space_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  d_local_hotness_list_ =
      buffer_ptr->reserve({h_local_hotness_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_local_ev_size_list_ =
      buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  d_local_ev_size_offset_ = buffer_ptr->reserve({h_local_ev_size_offset_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  d_local_combiner_list_ =
      buffer_ptr->reserve({h_local_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  d_network_embedding_list_ = buffer_ptr->reserve({h_network_embedding_list_.size()},
                                                  DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();

  d_local_embedding_list_.copy_from(h_local_embedding_list_);
  d_local_id_space_list_.copy_from(h_local_id_space_list_);
  d_local_hotness_list_.copy_from(h_local_hotness_list_);
  d_local_ev_size_list_.copy_from(h_local_ev_size_list_);
  d_local_ev_size_offset_.copy_from(h_local_ev_size_offset_);
  d_local_combiner_list_.copy_from(h_local_combiner_list_);
  d_network_embedding_list_.copy_from(h_network_embedding_list_);
}
}  // namespace embedding
