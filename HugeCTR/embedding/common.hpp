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
#include <core23/low_level_primitives.hpp>
#include <core23/registry.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <core23/tensor_params.hpp>
#include <map>
#include <string>
#include <vector>

namespace HugeCTR {

namespace core23 {

template <typename BuiltInType>
core23::Tensor init_tensor_list(int64_t n, int device_id) {
  core23::Device device(core23::DeviceType::GPU, device_id);
  core23::TensorParams params = core23::TensorParams().device(device);
  static_assert(sizeof(void *) == sizeof(BuiltInType *));
  constexpr int64_t pointer_width = sizeof(void *) / sizeof(BuiltInType);

  return core23::Tensor(
      params.shape({n, pointer_width}).data_type(core23::ToScalarType<BuiltInType>::value));
}

template <typename BuiltInType>
core23::Tensor init_tensor_list(const std::vector<core23::Tensor> &tensor_vec, int device_id,
                                cudaStream_t stream = 0) {
  std::vector<BuiltInType *> data_vec;
  for (auto &tensor : tensor_vec) {
    data_vec.push_back(tensor.data<BuiltInType>());
  }

  core23::Device device(core23::DeviceType::GPU, device_id);
  core23::TensorParams params = core23::TensorParams().device(device);
  static_assert(sizeof(void *) == sizeof(BuiltInType *));
  constexpr int64_t pointer_width = sizeof(void *) / sizeof(BuiltInType);
  auto tensor_list =
      core23::Tensor(params.shape({static_cast<int64_t>(data_vec.size()), pointer_width})
                         .data_type(core23::ToScalarType<BuiltInType>::value));
  if (stream != 0) {
    HCTR_LIB_THROW(cudaMemcpyAsync(tensor_list.data(), data_vec.data(),
                                   data_vec.size() * sizeof(BuiltInType *), cudaMemcpyHostToDevice,
                                   stream));
  } else {
    HCTR_LIB_THROW(cudaMemcpy(tensor_list.data(), data_vec.data(),
                              data_vec.size() * sizeof(BuiltInType *), cudaMemcpyHostToDevice));
  }

  return tensor_list;
}
}  // namespace core23
}  // namespace HugeCTR

namespace embedding {
namespace core23 = HugeCTR::core23;
using core::CoreResourceManager;

#define ALL_DATA_TYPES_CONVERTED_CORE_TO_CORE23()                   \
  {core::TensorScalarType::Float16, core23::ScalarType::Half},      \
      {core::TensorScalarType::Float32, core23::ScalarType::Float}, \
      {core::TensorScalarType::Char, core23::ScalarType::Char},     \
      {core::TensorScalarType::Int32, core23::ScalarType::Int32},   \
      {core::TensorScalarType::UInt32, core23::ScalarType::UInt32}, \
      {core::TensorScalarType::Int64, core23::ScalarType::Int64}, { \
    core::TensorScalarType::UInt64, core23::ScalarType::UInt64      \
  }

static std::unordered_map<core::TensorScalarType, core23::ScalarType> data_type_core_to_core23{
    ALL_DATA_TYPES_CONVERTED_CORE_TO_CORE23()};

#define ALL_DATA_TYPES_CONVERTED_CORE23_TO_CORE()                   \
  {core23::ScalarType::Half, core::TensorScalarType::Float16},      \
      {core23::ScalarType::Float, core::TensorScalarType::Float32}, \
      {core23::ScalarType::Char, core::TensorScalarType::Char},     \
      {core23::ScalarType::Int32, core::TensorScalarType::Int32},   \
      {core23::ScalarType::UInt32, core::TensorScalarType::UInt32}, \
      {core23::ScalarType::Int64, core::TensorScalarType::Int64}, { \
    core23::ScalarType::UInt64, core::TensorScalarType::UInt64      \
  }

static std::unordered_map<core23::ScalarType, core::TensorScalarType> data_type_core23_to_core{
    ALL_DATA_TYPES_CONVERTED_CORE23_TO_CORE()};

core::Tensor convert_core23_tensor_to_core_tensor(core23::Tensor native_tensor);
core23::Tensor convert_core_tensor_to_core23_tensor(core::Tensor native_tensor);

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
  core23::DataType key_type;
  core23::DataType index_type;
  core23::DataType offset_type;
  core23::DataType emb_type;

  EmbeddingLayout input_layout_;   // Only work in HugeCTR, specified the input layout.
  EmbeddingLayout output_layout_;  // Only work in HugeCTR, specifies the output layout.
  bool indices_only_;
  EmbeddingCollectionParam(int num_table, const std::vector<int> &table_id_to_vocabulary_size,
                           int num_lookup, const std::vector<LookupParam> &lookup_params,
                           const std::vector<std::vector<int>> &shard_matrix,
                           const std::vector<GroupedEmbeddingParam> &grouped_emb_params,
                           int universal_batch_size, core23::DataType key_type,
                           core23::DataType index_type, core23::DataType offset_type,
                           core23::DataType emb_type, EmbeddingLayout input_layout_,
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
  core23::DataType type;
};

struct EmbeddingInput {
  core23::Tensor keys;
  core23::Tensor num_keys;  // TODO: move from cpu to gpu
  core23::Tensor bucket_range;
  size_t h_num_keys;  // TODO: remove
  core23::Tensor fullbatch_bucket_range;
};

struct EmbeddingOutputAttr : public EVBufferAttr {
  core23::Tensor id_to_ev_size;
  core23::Tensor id_to_ev_start_indices;
  core23::Tensor id_to_combiner;
  int num_elements_per_sample;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);
};

struct EmbeddingOutput {
  core23::Tensor data;
  EmbeddingOutputAttr attr;
};

struct WgradAttr {
  int num_table;
  int num_lookup;
  core23::Tensor lookup_id_to_table_ids;
  core23::Tensor sorted_lookup_ids;
  core23::Tensor sorted_table_ids;
  core23::Tensor sorted_unique_table_ids;
  core23::Tensor table_id_to_ev_size;

  std::vector<int> h_sorted_unique_table_ids;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id);

  const core23::Tensor &get_unique_table_ids() const {
    return (num_table == num_lookup) ? lookup_id_to_table_ids : sorted_unique_table_ids;
  }
};

struct Wgrad {
  WgradAttr attr;

  core23::Tensor unique_keys;
  core23::Tensor num_unique_keys;
  core23::Tensor table_ids;
  core23::Tensor ev_start_indices;

  core23::Tensor table_range;

  core23::Tensor data;
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
