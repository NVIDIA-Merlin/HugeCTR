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
#pragma once
#include "common.hpp"

namespace embedding {
using HugeCTR::CudaDeviceContext;

class GlobalEmbeddingData {
 public:
  std::shared_ptr<CoreResourceManager> core_;

  int num_embedding_;
  int num_hotness_;

  std::vector<int> h_hotness_list_;
  std::vector<int> h_ev_size_list_;
  std::vector<int> h_ev_size_offset_;
  std::vector<char> h_combiner_list_;

  Tensor d_hotness_list_;
  Tensor d_ev_size_list_;
  Tensor d_ev_size_offset_;
  Tensor d_combiner_list_;

  GlobalEmbeddingData() = default;

  GlobalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                      const EmbeddingCollectionParam &params)
      : core_(core), num_embedding_(params.num_embedding), h_ev_size_offset_{0} {
    CudaDeviceContext context(core_->get_device_id());
    auto &embedding_params = params.embedding_params;

    for (int embedding_id = 0; embedding_id < num_embedding_; ++embedding_id) {
      h_hotness_list_.push_back(embedding_params[embedding_id].hotness);
      h_ev_size_list_.push_back(embedding_params[embedding_id].ev_size);
      h_combiner_list_.push_back(static_cast<char>(embedding_params[embedding_id].combiner));
    }
    std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                     std::back_inserter(h_ev_size_offset_));

    num_hotness_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);

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
};

class LocalEmbeddingData {
 public:
  std::shared_ptr<CoreResourceManager> core_;

  int num_local_embedding_;
  int num_local_hotness_;
  int sharding_id_;
  int num_sharding_;

  std::vector<int> h_local_embedding_list_;
  std::vector<std::vector<int>> h_global_embedding_list_;

  std::vector<int> h_local_id_space_list_;
  std::vector<int> h_local_hotness_list_;
  std::vector<int> h_local_ev_size_list_;
  std::vector<int> h_local_ev_size_offset_;
  std::vector<char> h_local_combiner_list_;

  Tensor d_local_embedding_list_;
  Tensor d_local_id_space_list_;
  Tensor d_local_hotness_list_;
  Tensor d_local_ev_size_list_;
  Tensor d_local_ev_size_offset_;
  Tensor d_local_combiner_list_;

  LocalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                     const EmbeddingCollectionParam &params,
                     const EmbeddingShardingParam &sharding_param)
      : core_(core),
        num_local_embedding_(sharding_param.local_embedding_list.size()),
        sharding_id_(sharding_param.sharding_id),
        num_sharding_(sharding_param.num_sharding),
        h_local_ev_size_offset_{0} {
    CudaDeviceContext context(core_->get_device_id());
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

    num_local_hotness_ =
        std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);

    // init device bufffer
    auto buffer_ptr = GetBuffer(core_);
    d_local_embedding_list_ = buffer_ptr->reserve({h_local_embedding_list_.size()}, DeviceType::GPU,
                                                  TensorScalarType::Int32);
    d_local_id_space_list_ = buffer_ptr->reserve({h_local_id_space_list_.size()}, DeviceType::GPU,
                                                 TensorScalarType::Int32);
    d_local_hotness_list_ = buffer_ptr->reserve({h_local_hotness_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
    d_local_ev_size_list_ = buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
    d_local_ev_size_offset_ = buffer_ptr->reserve({h_local_ev_size_offset_.size()}, DeviceType::GPU,
                                                  TensorScalarType::Int32);
    d_local_combiner_list_ = buffer_ptr->reserve({h_local_combiner_list_.size()}, DeviceType::GPU,
                                                 TensorScalarType::Char);

    buffer_ptr->allocate();

    d_local_embedding_list_.copy_from(h_local_embedding_list_);
    d_local_id_space_list_.copy_from(h_local_id_space_list_);
    d_local_hotness_list_.copy_from(h_local_hotness_list_);
    d_local_ev_size_list_.copy_from(h_local_ev_size_list_);
    d_local_ev_size_offset_.copy_from(h_local_ev_size_offset_);
    d_local_combiner_list_.copy_from(h_local_combiner_list_);
  }
};
}  // namespace embedding
