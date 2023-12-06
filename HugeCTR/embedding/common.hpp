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

#include <core/core.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/registry.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_operations.hpp>
#include <core23/tensor_params.hpp>
#include <map>
#include <set>
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
  auto tensor_list = init_tensor_list<BuiltInType>(tensor_vec.size(), device_id);

  std::vector<BuiltInType *> data_vec;
  for (auto &tensor : tensor_vec) {
    data_vec.push_back(tensor.data<BuiltInType>());
  }

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

static void init_tensor_list(core23::Tensor &tensor_list,
                             const std::vector<core23::Tensor> &tensor_vec,
                             cudaStream_t stream = 0) {
  if (tensor_list.device() == Device(DeviceType::CPU)) {
    for (size_t i = 0; i < tensor_vec.size(); ++i) {
      tensor_list.data<void *>()[i] = tensor_vec[i].data();
    }
  } else {
    std::vector<void *> data_vec;
    for (auto &tensor : tensor_vec) {
      data_vec.push_back(tensor.data());
    }

    if (stream != 0) {
      HCTR_LIB_THROW(cudaMemcpyAsync(tensor_list.data(), data_vec.data(),
                                     data_vec.size() * sizeof(void *), cudaMemcpyHostToDevice,
                                     stream));
    } else {
      HCTR_LIB_THROW(cudaMemcpy(tensor_list.data(), data_vec.data(),
                                data_vec.size() * sizeof(void *), cudaMemcpyHostToDevice));
    }
  }
}

}  // namespace core23
}  // namespace HugeCTR

namespace HugeCTR {

struct DataDistributionInput {
  core23::Tensor h_ptrs_;
  core23::Tensor d_ptrs_;
  int num_lookup_;
  std::vector<int> dr_lookup_ids_;
  core23::DataType key_type;
  core23::DataType offset_type;

  DataDistributionInput() = default;

  DataDistributionInput(std::shared_ptr<core::CoreResourceManager> core,
                        const std::vector<int> &dr_lookup_ids, core23::DataType key_type,
                        core23::DataType offset_type);

  void copy_tensor_vec(const std::vector<core23::Tensor> &dp_keys,
                       const std::vector<core23::Tensor> &dp_bucket_range, cudaStream_t stream);

  template <typename KeyType>
  const KeyType **get_dp_keys_pointer_ptr() const {
    return (const KeyType **)d_ptrs_.data();
  }

  template <typename BucketRangeType>
  const BucketRangeType **get_dp_bucket_range_pointer_ptr() const {
    return (const BucketRangeType **)(d_ptrs_.data()) + num_lookup_;
  }
};
}  // namespace HugeCTR

namespace embedding {
namespace core23 = HugeCTR::core23;
using core::CoreResourceManager;

// enum class which means pooling operation after lookup. Can be Sum, Average, Concat
enum class Combiner : char { Sum, Average, Concat };
std::ostream &operator<<(std::ostream &os, const Combiner &p);

enum class TablePlacementStrategy : int8_t {
  DataParallel,
  ModelParallel,
};
enum class CompressionStrategy : int8_t { Reduction, Unique };
std::ostream &operator<<(std::ostream &os, const CompressionStrategy &p);
enum class EmbeddingLayout : int8_t { FeatureMajor, BatchMajor };
std::ostream &operator<<(std::ostream &os, const EmbeddingLayout &p);
enum class CommunicationStrategy : int8_t { Uniform, Hierarchical };
std::ostream &operator<<(std::ostream &os, const CommunicationStrategy &p);
enum class SortStrategy : int8_t { Radix, Segmented };
std::ostream &operator<<(std::ostream &os, const SortStrategy &p);
enum class KeysPreprocessStrategy : int8_t { None, AddOffset };
std::ostream &operator<<(std::ostream &os, const KeysPreprocessStrategy &p);
enum class AllreduceStrategy : int8_t { Dense, GroupDense };
std::ostream &operator<<(std::ostream &os, const AllreduceStrategy &p);
enum class EmbeddingGroupType : int8_t {
  DataParallel,
  DenseModelParallel,
  SparseModelParallel,
  DenseModelParallelWithReduction,
};

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

struct GroupedTableParam {
  TablePlacementStrategy table_placement_strategy;
  std::vector<int> table_ids;

  GroupedTableParam(TablePlacementStrategy _table_placement_strategy,
                    const std::vector<int> &_table_ids)
      : table_placement_strategy(_table_placement_strategy), table_ids(_table_ids) {}
};

struct CompressionParam {
  std::unordered_map<CompressionStrategy, std::set<int>> compression_strategy_to_table_ids;
};
std::ostream &operator<<(std::ostream &os, const CompressionParam &p);

struct GroupedLookupParam {
  int grouped_table_idx;
  std::vector<int> lookup_ids;
  EmbeddingGroupType embedding_group_type;

  GroupedLookupParam(int table_idx, const std::vector<int> &lookup_ids,
                     EmbeddingGroupType embedding_group_type)
      : grouped_table_idx(table_idx),
        lookup_ids(lookup_ids),
        embedding_group_type(embedding_group_type) {}
};

std::vector<int> filter_dp_lookup_ids(const std::vector<LookupParam> &lookup_params,
                                      const GroupedTableParam &table_param);

std::vector<int> filter_mp_sparse_lookup_ids(const std::vector<LookupParam> &lookup_params,
                                             const GroupedTableParam &table_param);

std::vector<int> filter_mp_dense_lookup_ids(const std::vector<LookupParam> &lookup_params,
                                            const GroupedTableParam &table_param);

struct EmbeddingCollectionParam {
  int num_table;

  int num_lookup;
  std::vector<LookupParam> lookup_params;  // num of lookup

  std::vector<std::vector<int>> shard_matrix;  // num_gpus * num_table
  std::vector<GroupedTableParam> grouped_table_params;
  std::vector<GroupedLookupParam> grouped_lookup_params;

  int universal_batch_size;
  core23::DataType key_type;
  core23::DataType index_type;
  core23::DataType offset_type;
  core23::DataType emb_type;
  core23::DataType wgrad_type_;

  EmbeddingLayout input_layout_;   // Only work in HugeCTR, specified the input layout.
  EmbeddingLayout output_layout_;  // Only work in HugeCTR, specifies the output layout.

  SortStrategy sort_strategy_;
  KeysPreprocessStrategy keys_preprocess_strategy_;
  AllreduceStrategy allreduce_strategy_;
  CommunicationStrategy comm_strategy_;

  EmbeddingCollectionParam(
      int num_table, int num_lookup, const std::vector<LookupParam> &lookup_params,
      const std::vector<std::vector<int>> &shard_matrix,
      const std::vector<GroupedTableParam> &grouped_table_params, int universal_batch_size,
      core23::DataType key_type, core23::DataType index_type, core23::DataType offset_type,
      core23::DataType emb_type, core23::DataType wgrad_type, EmbeddingLayout input_layout_,
      EmbeddingLayout output_layout, SortStrategy sort_strategy,
      KeysPreprocessStrategy keys_preprocess_strategy, AllreduceStrategy allreduce_strategy,
      CommunicationStrategy comm_strategy, CompressionParam compreesion_param);

  bool lookup_id_in_group(size_t grouped_id, int lookup_id) const {
    const auto &group_param = this->grouped_lookup_params[grouped_id];
    return std::find(group_param.lookup_ids.begin(), group_param.lookup_ids.end(), lookup_id) !=
           group_param.lookup_ids.end();
  }

  bool has_table_shard(int gpu_id, size_t grouped_id, int lookup_id) const {
    int table_id = this->lookup_params[lookup_id].table_id;
    bool has_portion = (this->shard_matrix[gpu_id][table_id] != 0);
    return this->lookup_id_in_group(grouped_id, lookup_id) && has_portion;
  }

  void get_table_shard_id(int gpu_id, int table_id, int *shard_id, int *num_shard) const {
    size_t num_gpus = shard_matrix.size();

    std::vector<int> shard_gpus;
    for (size_t ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
      if (this->shard_matrix[ggpu_id][table_id] == 1) {
        shard_gpus.push_back(ggpu_id);
      }
    }
    auto find_shard_id_iter = std::find(shard_gpus.begin(), shard_gpus.end(), gpu_id);
    HCTR_CHECK_HINT(find_shard_id_iter != shard_gpus.end(),
                    "get_table_shard_id does not find shard id");
    *shard_id = std::distance(shard_gpus.begin(), find_shard_id_iter);
    *num_shard = static_cast<int>(shard_gpus.size());
  }
};

struct EmbeddingInput {
  core23::Tensor keys;
  core23::Tensor num_keys;  // TODO: move from cpu to gpu
  size_t h_num_keys;        // TODO: remove

  core23::Tensor bucket_range;
  core23::Tensor num_keys_per_bucket;

  struct DenseCompressionInput {
    struct ModelParallelCompressionInput {
      core23::Tensor h_send_k_per_gpu;  // bucket_range_type
      core23::Tensor h_recv_k_per_gpu;  // bucket_range_type

      core23::Tensor model_reverse_idx;  // bucket_range_type
      size_t num_model_reverse_idx;
      core23::Tensor network_reverse_idx;  // bucket_range_type
      size_t num_network_reverse_idx;
      core23::Tensor network_dst_bucket_ids;  // bucket_range_type
    } model_parallel_compression_input;

    core23::Tensor num_keys_per_table_offset;  // bucket_range_type
    core23::Tensor table_ids;                  //  int
  } dense_compression_input;
};

struct EmbeddingOutputAttr {
  mutable std::vector<int> h_id_to_hotness;
  mutable int hotness_sum;

  int num_lookup;

  std::vector<int> h_id_to_ev_size;
  std::vector<char> h_id_to_combiner;
  std::vector<int> h_id_to_ev_start_indices{0};

  core23::Tensor id_to_ev_size;
  core23::Tensor id_to_ev_start_indices;
  core23::Tensor id_to_combiner;
  int num_elements_per_sample;

  EmbeddingLayout layout;
  int max_ev_size;
  bool is_ragged;
  bool is_aligned;
  core23::DataType type;

  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);

  void update_mutable_data(std::shared_ptr<CoreResourceManager> core,
                           const EmbeddingCollectionParam &ebc_param) const;
};

struct EmbeddingOutput {
  core23::Tensor data;
  EmbeddingOutputAttr attr;
};

// this struct is used in data distributor because this struct stores mapping needed for calculating
// the backwward index calculation.
struct WgradAttr {
  int num_table;
  int num_lookup;
  core23::Tensor lookup_id_to_table_ids;
  core23::Tensor sorted_lookup_ids;
  core23::Tensor sorted_table_ids;
  core23::Tensor sorted_unique_table_ids;
  core23::Tensor table_id_to_ev_size;
  core23::DataType type;

  std::vector<int> h_sorted_unique_table_ids;

  bool is_same_ev_size = false;
  int same_ev_size = 0;
  void init(std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
            size_t grouped_id);

  const core23::Tensor &get_unique_table_ids() const {
    return (num_table == num_lookup) ? lookup_id_to_table_ids : sorted_unique_table_ids;
  }
};

struct Wgrad {
  WgradAttr attr;

  core23::Tensor unique_keys;
  core23::Tensor num_unique_keys;   // uint64_t
  core23::Tensor table_ids;         // int
  core23::Tensor ev_start_indices;  // uint32_t

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
  std::vector<int> table_id_to_vocabulary_size;
  size_t grouped_id;
  WgradAttr wgrad_attr;

  Wgrad *wgrad = nullptr;
  AllreduceWgradInitializer &init(Wgrad &other);

  AllreduceWgradInitializer &init_indices();
  AllreduceWgradInitializer &init_data(bool grouped, const core23::BufferChannel &buffer_channel);
};

double get_dense_unique_ratio();

double get_wgrad_unique_ratio();
}  // namespace embedding
