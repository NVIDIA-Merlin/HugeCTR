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

class GlobalEmbeddingData {
 public:
  std::shared_ptr<CoreResourceManager> core_;

  int num_embedding_;
  int num_hotness_;
  int max_ev_size_;

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
                      const EmbeddingCollectionParam &params);
};

class LocalEmbeddingData {
 public:
  std::shared_ptr<CoreResourceManager> core_;

  int num_local_embedding_;
  int num_local_hotness_;
  int shard_id_;
  int shards_count_;
  int max_ev_size_;

  std::vector<int> h_local_embedding_list_;
  std::vector<std::vector<int>> h_global_embedding_list_;

  std::vector<int> h_local_id_space_list_;
  std::vector<int> h_local_hotness_list_;
  std::vector<int> h_local_ev_size_list_;
  std::vector<int> h_local_ev_size_offset_;
  std::vector<char> h_local_combiner_list_;
  std::vector<int> h_network_embedding_list_;
  std::vector<char> h_network_combiner_list_;

  Tensor d_local_embedding_list_;
  Tensor d_local_id_space_list_;
  Tensor d_local_hotness_list_;
  Tensor d_local_ev_size_list_;
  Tensor d_local_ev_size_offset_;
  Tensor d_local_combiner_list_;
  Tensor d_network_embedding_list_;

  LocalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                     const EmbeddingCollectionParam &params,
                     const EmbeddingShardingParam &sharding_param);

  LocalEmbeddingData(std::shared_ptr<CoreResourceManager> core,
                     const EmbeddingCollectionParam &params,
                     const EmbeddingShardParam &shard_param);
};
}  // namespace embedding
