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
#pragma once

#include <core/buffer.hpp>
#include <map>
#include <string>
#include <vector>

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
std::ostream &operator<<(std::ostream &os, const EmbeddingLayout &p);

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
  std::vector<int> table_id_to_vocabulary_size;

  int num_lookup;
  std::vector<LookupParam> lookup_params;  // num of lookup

  std::vector<std::vector<int>> shard_matrix;  // num_gpus * num_table
  std::vector<GroupedEmbeddingParam> grouped_emb_params;

  int universal_batch_size;
  DataType key_type;
  DataType index_type;
  DataType offset_type;
  DataType emb_type;

  EmbeddingLayout input_layout_;   // Only work in HugeCTR, specified the input layout.
  EmbeddingLayout output_layout_;  // Only work in HugeCTR, specifies the output layout.
  bool indices_only_;
  EmbeddingCollectionParam(int num_table, const std::vector<int> &table_id_to_vocabulary_size,
                           int num_lookup, const std::vector<LookupParam> &lookup_params,
                           const std::vector<std::vector<int>> &shard_matrix,
                           const std::vector<GroupedEmbeddingParam> &grouped_emb_params,
                           int universal_batch_size, DataType key_type, DataType index_type,
                           DataType offset_type, DataType emb_type, EmbeddingLayout input_layout_,
                           EmbeddingLayout output_layout, bool indices_only)
      : num_table(num_table),
        table_id_to_vocabulary_size(table_id_to_vocabulary_size),
        num_lookup(num_lookup),
        lookup_params(lookup_params),
        shard_matrix(shard_matrix),
        grouped_emb_params(grouped_emb_params),
        universal_batch_size(universal_batch_size),
        key_type(key_type),
        index_type(index_type),
        offset_type(offset_type),
        emb_type(emb_type),
        input_layout_(input_layout_),
        output_layout_(output_layout),
        indices_only_(indices_only) {}

  bool lookup_id_in_group(size_t grouped_id, int lookup_id) const {
    const auto &group_param = this->grouped_emb_params[grouped_id];
    int table_id = this->lookup_params[lookup_id].table_id;
    return std::find(group_param.table_ids.begin(), group_param.table_ids.end(), table_id) !=
           group_param.table_ids.end();
  }

  bool has_table_shard(int gpu_id, size_t grouped_id, int lookup_id) const {
    int table_id = this->lookup_params[lookup_id].table_id;
    bool has_portion = (this->shard_matrix[gpu_id][table_id] != 0);
    return this->lookup_id_in_group(grouped_id, lookup_id) && has_portion;
  }

  std::vector<int> get_table_id_to_global_start_indices() const;
};

struct KernelParams {
  int num_sms;

  void init();
};

struct EVBufferAttr {
  EmbeddingLayout layout;
  int max_ev_size;
  bool is_ragged;
  bool is_aligned;
  DataType type;
};

struct EmbeddingInput {
  Tensor keys;
  Tensor num_keys;  // TODO: move from cpu to gpu
  Tensor bucket_range;
  size_t h_num_keys;  // TODO: remove
  Tensor fullbatch_bucket_range;
};

struct EmbeddingOutputAttr : public EVBufferAttr {
  Tensor id_to_ev_size;
  Tensor id_to_ev_start_indices;
  Tensor id_to_combiner;
  int num_elements_per_sample;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);
};

struct EmbeddingOutput {
  Tensor data;
  EmbeddingOutputAttr attr;
};

struct WgradAttr {
  int num_table;
  int num_lookup;
  Tensor lookup_id_to_table_ids;
  Tensor sorted_lookup_ids;
  Tensor sorted_table_ids;
  Tensor sorted_unique_table_ids;
  Tensor table_id_to_ev_size;

  std::vector<int> h_sorted_unique_table_ids;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id);

  const Tensor &get_unique_table_ids() const {
    return (num_table == num_lookup) ? lookup_id_to_table_ids : sorted_unique_table_ids;
  }
};

struct Wgrad {
  WgradAttr attr;

  Tensor unique_keys;
  Tensor num_unique_keys;
  Tensor table_ids;
  Tensor ev_start_indices;

  Tensor table_range;

  Tensor data;
};

struct WgradInitializer {
  std::shared_ptr<CoreResourceManager> core;
  EmbeddingCollectionParam ebc_param;
  size_t grouped_id;
  WgradAttr wgrad_attr;

  Wgrad *wgrad = nullptr;
  WgradInitializer &init(Wgrad &other);

  WgradInitializer &init_indices();

  WgradInitializer &init_data();
};

struct AllreduceWgradInitializer {
  std::shared_ptr<CoreResourceManager> core;
  EmbeddingCollectionParam ebc_param;
  size_t grouped_id;
  WgradAttr wgrad_attr;

  Wgrad *wgrad = nullptr;
  AllreduceWgradInitializer &init(Wgrad &other);

  AllreduceWgradInitializer &init_indices();

  AllreduceWgradInitializer &init_data();
};

}  // namespace embedding
