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
#include "HugeCTR/include/optimizer.hpp"

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
using HugeCTR::TensorScalarType;

// enum class which means pooling operation after lookup. Can be Sum, Average, Concat
enum class Combiner : char { Sum, Average, Concat };
std::ostream &operator<<(std::ostream &os, const Combiner &p);

enum class TablePlacementStrategy : int8_t {
  DataParallel,
  Localized
  // ColumnwiseDistributed,
  // RowwiseDistributed,
  // RowwiseDistributedWithPacking,
  // All2AllDense
};

const std::map<std::string, TablePlacementStrategy> _table_placement_type_map = {
    {"dp", TablePlacementStrategy::DataParallel}, {"localized", TablePlacementStrategy::Localized}};

struct EmbeddingShardingParam {
  std::vector<int> local_embedding_list;
  std::vector<std::vector<int>> global_embedding_list;
  int sharding_id;
  int num_sharding;
  TablePlacementStrategy table_placement_strategy;
  // there will be more. for example, FrequentEmbedding may need num_frequent_categories.
};

struct EmbeddingParam {
  int embedding_id;
  int id_space;
  Combiner combiner;
  int hotness;
  int ev_size;
};
std::ostream &operator<<(std::ostream &os, const EmbeddingParam &p);

struct EmbeddingCollectionParam {
  int num_embedding;
  std::vector<EmbeddingParam> embedding_params;  // num of bucket

  int universal_batch_size;
  DataType key_type;
  DataType index_type;
  DataType offset_type;
  DataType emb_type;
  bool fuse_lookup = false;
  bool fuse_communication = false;
  bool is_dataparall_input = false;
  bool is_table_first_input = false;
  bool is_utest = false;
};

// we want to treat concat embedding as several seperate embedding
void flatten_concat_embedding(EmbeddingCollectionParam *_ebc_param,
                              std::vector<std::vector<EmbeddingShardingParam>> *_ebs_param);
}  // namespace embedding