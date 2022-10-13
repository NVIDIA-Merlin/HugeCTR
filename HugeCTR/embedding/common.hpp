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
#include <map>
#include <string>
#include <vector>

#include "HugeCTR/core/buffer.hpp"

namespace embedding {
using core::CoreResourceManager;
using core::DataType;
using core::Device;
using core::DeviceType;
using core::GetBuffer;
using core::GetBufferBlock;
using core::Shape;
using core::Tensor;
using core::TensorList;
using core::TensorScalarType;

// enum class which means pooling operation after lookup. Can be Sum, Average, Concat
enum class Combiner : char { Sum, Average, Concat };
std::ostream &operator<<(std::ostream &os, const Combiner &p);

enum class TablePlacementStrategy : int8_t { DataParallel, ModelParallel, Hybrid };
enum class EmbeddingLayout : int8_t { FeatureMajor, BatchMajor };

const std::map<std::string, TablePlacementStrategy> _table_placement_type_map = {
    {"dp", TablePlacementStrategy::DataParallel},
    {"mp", TablePlacementStrategy::ModelParallel},
    {"hybrid", TablePlacementStrategy::Hybrid}};

struct LookupParam {
  int lookup_id;
  int table_id;
  Combiner combiner;
  int max_hotness;
  int ev_size;

  LookupParam(int lookup_id, int table_id, Combiner combiner, int max_hotness, int ev_size)
      : lookup_id(lookup_id),
        table_id(table_id),
        combiner(combiner),
        max_hotness(max_hotness),
        ev_size(ev_size) {}
};
std::ostream &operator<<(std::ostream &os, const LookupParam &p);

struct GroupedEmbeddingParam {
  TablePlacementStrategy table_placement_strategy;
  std::vector<int> table_ids;

  GroupedEmbeddingParam(TablePlacementStrategy _table_placement_strategy,
                        const std::vector<int> &_table_ids)
      : table_placement_strategy(_table_placement_strategy), table_ids(_table_ids) {}
};

struct EmbeddingCollectionParam {
  int num_table;
  int num_lookup;
  std::vector<LookupParam> lookup_params;  // num of lookup

  std::vector<std::vector<int>> shard_matrix;  // num_gpus * num_table
  std::vector<GroupedEmbeddingParam> grouped_emb_params;

  int universal_batch_size;
  DataType key_type;
  DataType index_type;
  DataType offset_type;
  DataType emb_type;
  EmbeddingLayout input_layout_;

  EmbeddingCollectionParam(int num_table, int num_lookup,
                           const std::vector<LookupParam> &lookup_params,
                           const std::vector<std::vector<int>> &shard_matrix,
                           const std::vector<GroupedEmbeddingParam> &grouped_emb_params,
                           int universal_batch_size, DataType key_type, DataType index_type,
                           DataType offset_type, DataType emb_type, EmbeddingLayout input_layout_)
      : num_table(num_table),
        num_lookup(num_lookup),
        lookup_params(lookup_params),
        shard_matrix(shard_matrix),
        grouped_emb_params(grouped_emb_params),
        universal_batch_size(universal_batch_size),
        key_type(key_type),
        index_type(index_type),
        offset_type(offset_type),
        emb_type(emb_type),
        input_layout_(input_layout_) {}
};

struct UniformModelParallelEmbeddingMeta {
  mutable std::vector<int> h_hotness_list_;
  mutable int hotness_sum_;
  mutable std::vector<int> h_local_hotness_list_;
  mutable int num_local_hotness_;

  int num_lookup_;

  std::vector<int> h_ev_size_list_;
  std::vector<int> h_ev_size_offset_;
  Tensor d_ev_size_offset_;
  int max_ev_size_;
  int num_sms_;

  std::vector<char> h_combiner_list_;
  Tensor d_combiner_list_;

  int num_local_lookup_;
  std::vector<int> h_local_shard_id_list_;
  Tensor d_local_shard_id_list_;

  std::vector<int> h_local_num_shards_list_;
  Tensor d_local_num_shards_list_;

  std::vector<int> h_local_table_id_list_;
  Tensor d_local_table_id_list_;

  std::vector<int> h_local_lookup_id_list_;
  Tensor d_local_lookup_id_list_;

  std::vector<int> h_local_ev_size_list_;
  Tensor d_local_ev_size_list_;

  std::vector<int> h_local_ev_size_offset_;
  Tensor d_local_ev_size_offset_;

  std::vector<std::vector<int>> h_global_lookup_id_list_;

  std::vector<int> h_network_lookup_id_list_;
  Tensor d_network_lookup_id_list_;

  std::vector<char> h_network_combiner_list_;

  std::vector<int> h_network_ids_;
  Tensor network_ids_;

  std::vector<int> h_network_gpu_ids_;
  Tensor network_gpu_ids_;

  std::vector<int> h_network_offsets_;
  Tensor network_offsets_;

  std::vector<int> h_network_dst_lookup_ids_;
  Tensor network_dst_lookup_ids_;

  std::vector<std::vector<int>> h_network_ev_sizes_;
  std::vector<Tensor> network_ev_size_list_;
  TensorList network_ev_sizes_;

  std::vector<std::vector<int>> h_network_ev_offsets_;
  std::vector<Tensor> network_ev_offset_list_;
  TensorList network_ev_offsets_;

  UniformModelParallelEmbeddingMeta(std::shared_ptr<CoreResourceManager> core,
                                    const EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void update_mutable_meta(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param, size_t grouped_id) const;
};

struct UniformDataParallelEmbeddingMeta {
  mutable std::vector<int> h_hotness_list_;
  mutable int num_hotness_;
  mutable std::vector<int> h_local_hotness_list_;
  mutable int num_local_hotness_;

  int num_lookup_;
  std::vector<int> h_ev_size_list_;
  int max_ev_size_;
  std::vector<int> h_ev_size_offset_;
  Tensor d_ev_size_offset_;

  std::vector<char> h_combiner_list_;
  Tensor d_combiner_list_;

  int num_local_lookup_;

  std::vector<char> h_local_combiner_list_;

  std::vector<int> h_local_lookup_id_list_;
  Tensor d_local_lookup_id_list_;

  std::vector<int> h_local_ev_size_list_;
  Tensor d_local_ev_size_list_;

  std::vector<int> h_local_table_id_list_;
  Tensor d_local_table_id_list_;

  UniformDataParallelEmbeddingMeta(std::shared_ptr<CoreResourceManager> core,
                                   const EmbeddingCollectionParam &ebc_param, size_t grouped_id);

  void update_mutable_meta(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param, size_t grouped_id) const;
};
}  // namespace embedding
