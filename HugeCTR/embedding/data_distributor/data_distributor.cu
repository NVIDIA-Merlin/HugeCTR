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

#include <HugeCTR/include/utils.cuh>
#include <embedding/data_distributor/data_compression_operators.cuh>
#include <embedding/data_distributor/data_distributor.hpp>
#include <embedding/operators/communication.hpp>
#include <unordered_set>

namespace HugeCTR {

DataDistributor::DataDistributor(
    std::vector<std::shared_ptr<core::CoreResourceManager>>& core_resource_managers,
    const embedding::EmbeddingCollectionParam& ebc_param,
    const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list,
    const std::vector<int>& dr_lookup_ids)
    : core_resource_managers_(core_resource_managers),
      batch_size_(ebc_param.universal_batch_size),
      batch_size_per_gpu_(ebc_param.universal_batch_size /
                          core_resource_managers[0]->get_global_gpu_count()),
      ebc_param_(ebc_param),
      emb_table_param_list_(emb_table_param_list),
      num_local_gpus_(core_resource_managers[0]->get_local_gpu_count()),
      num_global_gpus_(core_resource_managers[0]->get_global_gpu_count()),
      num_features_(ebc_param.num_lookup) {
  resident_feature_tables_ = ebc_param.shard_matrix;

  // construct lookup mappings
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    const embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];
    feature_pooling_factors_.push_back(lookup_param.max_hotness);
    feature_id_to_table_id_map_[lookup_id] = lookup_param.table_id;
    for (size_t group_id = 0; group_id < ebc_param.grouped_lookup_params.size(); ++group_id) {
      if (!ebc_param.lookup_id_in_group(group_id, lookup_id)) continue;
      feature_id_to_group_id_map_[lookup_id] = group_id;
    }
  }

  init_comm_data();
  init_filtered_all_to_all();
  init_fixed_dp_bucket_range();

  for (size_t gpu_id = 0; gpu_id < num_local_gpus_; ++gpu_id) {
    data_distribution_input_.emplace_back(core_resource_managers_[gpu_id], dr_lookup_ids,
                                          ebc_param.key_type, ebc_param.offset_type);
  }
}

void DataDistributor::init_comm_data() {
  // Get number of features in each group
  size_t num_features = 0;
  for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
    const auto& lookup_param = ebc_param_.lookup_params[lookup_id];
    num_features += lookup_param.max_hotness;
  }

  for (size_t i = 0; i < num_local_gpus_; ++i) {
    CudaDeviceContext context(core_resource_managers_[i]->get_device_id());
    core23::Device device(core23::DeviceType::GPU, core_resource_managers_[i]->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    GpuCommData comm_data;
    comm_data.last_batch_size = 0;

    size_t num_keys = num_features * ebc_param_.universal_batch_size;

    comm_data.hotness_bucket_range =
        core23::Tensor(params.shape({static_cast<int64_t>(num_features_ + 1)})
                           .data_type(core23::ScalarType::Int32));

    std::vector<int> hotness_bucket_range(1, 0);
    std::copy(feature_pooling_factors_.begin(), feature_pooling_factors_.end(),
              back_inserter(hotness_bucket_range));
    std::inclusive_scan(hotness_bucket_range.begin() + 1, hotness_bucket_range.end(),
                        hotness_bucket_range.begin() + 1);

    core23::copy_sync(comm_data.hotness_bucket_range, hotness_bucket_range);

    gpu_comm_data_.emplace_back(comm_data);
  }
}

void DataDistributor::init_filtered_all_to_all() {
  // --- allocate operators ---
  for (size_t group_id = 0; group_id < ebc_param_.grouped_lookup_params.size(); group_id++) {
    std::vector<std::unique_ptr<IDataDistributionOp>> data_distribution_ops;

    auto embedding_group_type = ebc_param_.grouped_lookup_params[group_id].embedding_group_type;
    for (size_t i = 0; i < num_local_gpus_; ++i) {
      auto core = core_resource_managers_[i];
      CudaDeviceContext context(core->get_device_id());

      if (embedding_group_type == embedding::EmbeddingGroupType::DataParallel) {
        data_distribution_ops.push_back(std::make_unique<SparseDPDataDistributionOp>(
            core, ebc_param_, group_id, emb_table_param_list_));
      } else if (embedding_group_type == embedding::EmbeddingGroupType::SparseModelParallel) {
        data_distribution_ops.push_back(std::make_unique<SparseMPDataDistributionOp>(
            core, ebc_param_, group_id, emb_table_param_list_));
      } else if (embedding_group_type == embedding::EmbeddingGroupType::DenseModelParallel ||
                 embedding_group_type ==
                     embedding::EmbeddingGroupType::DenseModelParallelWithReduction) {
        data_distribution_ops.push_back(std::make_unique<DenseMPDataDistributionOp>(
            core, ebc_param_, group_id, emb_table_param_list_));
      } else {
        HCTR_OWN_THROW(Error_t::IllegalCall,
                       "table placement strategy and embedding type not compatible");
      }
    }
    data_distribution_ops_.push_back(std::move(data_distribution_ops));
  }
}

void DataDistributor::init_fixed_dp_bucket_range() {
  // ---- init static bucket range ----
  // TODO: remove when data reader returns bucket range
  fixed_dp_bucket_range_.resize(num_local_gpus_);

  for (size_t gpu_id = 0; gpu_id < num_local_gpus_; ++gpu_id) {
    auto core = core_resource_managers_[gpu_id];
    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::BufferParams buffer_params;
    buffer_params.unitary = false;
    core23::TensorParams params =
        core23::TensorParams().device(device).buffer_params(buffer_params);

    for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
      int num_buckets = batch_size_per_gpu_ + 1;

      auto bucket_range = core23::Tensor(
          params.shape({static_cast<int64_t>(num_buckets)}).data_type(ebc_param_.offset_type));

      fixed_dp_bucket_range_[gpu_id].push_back(std::move(bucket_range));
    }
  }

  // -- non-group specific operators
  for (size_t gpu_id = 0; gpu_id < num_local_gpus_; ++gpu_id) {
    compute_dp_bucket_range_operators_.emplace_back(core_resource_managers_[gpu_id], ebc_param_);
  }
}

void DataDistributor::distribute(int gpu_id, const std::vector<core23::Tensor>& dp_keys,
                                 const std::vector<core23::Tensor>& dp_bucket_range,
                                 DataDistributor::Result& output, int batch_size) {
  auto core = core_resource_managers_[gpu_id];
  CudaDeviceContext ctx(core->get_device_id());
  cudaStream_t stream = core->get_local_gpu()->get_stream();

  const bool bucket_ranges_outdated = batch_size != gpu_comm_data_[gpu_id].last_batch_size;
  gpu_comm_data_[gpu_id].last_batch_size = batch_size;

  // sparse_forward new full batch bucket range (to be deprecated)
  // sparse_forward dp bucket ranges (to be moved to data reader)
  if (bucket_ranges_outdated) {
    compute_dp_bucket_range_operators_[gpu_id](fixed_dp_bucket_range_[gpu_id],
                                               output[0].num_keys_per_bucket, batch_size, stream);

    // Instead of recomputing for each group, copy computed result
    for (size_t grouped_id = 1; grouped_id < ebc_param_.grouped_lookup_params.size();
         ++grouped_id) {
      HCTR_LIB_THROW(cudaMemcpyAsync(
          output[grouped_id].num_keys_per_bucket.data(), output[0].num_keys_per_bucket.data(),
          output[0].num_keys_per_bucket.num_bytes(), cudaMemcpyDeviceToDevice, stream));
    }
  }

  data_distribution_input_[gpu_id].copy_tensor_vec(dp_keys, fixed_dp_bucket_range_[gpu_id], stream);

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size(); grouped_id++) {
    data_distribution_ops_[grouped_id][gpu_id]->distribute(data_distribution_input_[gpu_id],
                                                           output[grouped_id], batch_size, stream);
  }
}

DataDistributor::Result allocate_output_for_data_distributor(
    std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param) {
  CudaDeviceContext context(core_resource_manager->get_device_id());
  int num_global_gpus = core_resource_manager->get_global_gpu_count();
  int batch_size = ebc_param.universal_batch_size;
  int batch_size_per_gpu = ebc_param.universal_batch_size / num_global_gpus;

  DataDistributor::Result output;
  for (size_t group_id = 0; group_id < ebc_param.grouped_lookup_params.size(); ++group_id) {
    auto& grouped_lookup_params = ebc_param.grouped_lookup_params[group_id];

    int batch_size_after_filter =
        grouped_lookup_params.embedding_group_type == embedding::EmbeddingGroupType::DataParallel
            ? batch_size_per_gpu
            : batch_size;
    size_t num_buckets = 0ul;
    size_t num_features = 0ul;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      if (!ebc_param.has_table_shard(core_resource_manager->get_global_gpu_id(), group_id,
                                     lookup_id)) {
        continue;
      }
      const auto& lookup_param = ebc_param.lookup_params[lookup_id];
      num_features += lookup_param.max_hotness;
      num_buckets += 1;
    }

    core23::Device device(core23::DeviceType::GPU, core_resource_manager->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    embedding::EmbeddingInput embedding_input;
    embedding_input.h_num_keys = 0ul;
    embedding_input.keys =
        core23::Tensor(params.shape({static_cast<int64_t>(batch_size_after_filter * num_features)})
                           .data_type(ebc_param.key_type));

    embedding_input.num_keys = core23::Tensor(
        params.shape({1}).data_type(core23::ScalarType::UInt64).device(core23::DeviceType::CPU));

    embedding_input.num_keys_per_bucket = core23::Tensor(
        params.shape({static_cast<int64_t>(batch_size_per_gpu * ebc_param.num_lookup)})
            .data_type(ebc_param.offset_type));

    if (grouped_lookup_params.embedding_group_type == embedding::EmbeddingGroupType::DataParallel ||
        grouped_lookup_params.embedding_group_type ==
            embedding::EmbeddingGroupType::SparseModelParallel) {
      embedding_input.bucket_range = core23::Tensor(
          params.shape({static_cast<int64_t>(batch_size_after_filter * num_buckets + 1)})
              .data_type(ebc_param.offset_type));
    } else if (grouped_lookup_params.embedding_group_type ==
                   embedding::EmbeddingGroupType::DenseModelParallel ||
               grouped_lookup_params.embedding_group_type ==
                   embedding::EmbeddingGroupType::DenseModelParallelWithReduction) {
      auto& dense_compression_input = embedding_input.dense_compression_input;
      embedding::WgradAttr wgrad_attr;
      wgrad_attr.init(core_resource_manager, ebc_param, group_id);
      dense_compression_input.num_keys_per_table_offset =
          core23::Tensor(params.shape({static_cast<int64_t>(wgrad_attr.num_table + 1)})
                             .data_type(ebc_param.offset_type));
      dense_compression_input.table_ids =
          core23::Tensor(params.shape({static_cast<int64_t>(wgrad_attr.num_table)})
                             .data_type(core23::ScalarType::Int32));

      auto& model_parallel_compression_input =
          dense_compression_input.model_parallel_compression_input;
      model_parallel_compression_input.h_send_k_per_gpu =
          core23::Tensor(params.shape({static_cast<int64_t>(num_global_gpus)})
                             .data_type(ebc_param.offset_type)
                             .device(core23::DeviceType::CPU));
      model_parallel_compression_input.h_recv_k_per_gpu =
          core23::Tensor(params.shape({static_cast<int64_t>(num_global_gpus)})
                             .data_type(ebc_param.offset_type)
                             .device(core23::DeviceType::CPU));

      model_parallel_compression_input.model_reverse_idx =
          core23::Tensor(params.shape({static_cast<int64_t>(batch_size * num_features)})
                             .data_type(ebc_param.offset_type));
      model_parallel_compression_input.num_model_reverse_idx = 0ul;

      size_t num_features_this_group_have = 0ul;
      for (int peer_gpu_id = 0; peer_gpu_id < num_global_gpus; ++peer_gpu_id) {
        for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
          if (!ebc_param.has_table_shard(peer_gpu_id, group_id, lookup_id)) {
            continue;
          }
          const auto& lookup_param = ebc_param.lookup_params[lookup_id];
          num_features_this_group_have += lookup_param.max_hotness;
        }
      }

      model_parallel_compression_input.network_reverse_idx = core23::Tensor(
          params.shape({static_cast<int64_t>(batch_size_per_gpu * num_features_this_group_have)})
              .data_type(ebc_param.offset_type));
      model_parallel_compression_input.num_network_reverse_idx = 0ul;
      model_parallel_compression_input.network_dst_bucket_ids = core23::Tensor(
          params.shape({static_cast<int64_t>(batch_size_per_gpu * num_features_this_group_have)})
              .data_type(ebc_param.offset_type));

      // initialize table_ids
      core23::copy_sync(dense_compression_input.table_ids, wgrad_attr.sorted_unique_table_ids);
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "not supported embedding_group_type.");
    }
    output.push_back(embedding_input);
  }
  return output;
}
}  // namespace HugeCTR
