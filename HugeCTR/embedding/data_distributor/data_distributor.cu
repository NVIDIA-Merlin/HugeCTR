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

#include <HugeCTR/embedding/common.hpp>
#include <HugeCTR/include/utils.cuh>
#include <core/hctr_impl/hctr_backend.hpp>
#include <cub/cub.cuh>
#include <embedding/data_distributor/data_distributor.hpp>
#include <embedding/data_distributor/gpu_kernels.hpp>
#include <embedding/operators/communication.hpp>

namespace HugeCTR {

DataDistributor::DataDistributor(
    size_t batch_size, core23::DataType scalar_type,
    std::shared_ptr<ResourceManager> resource_manager,
    std::vector<std::shared_ptr<core::CoreResourceManager>>& core_resource_managers,
    const embedding::EmbeddingCollectionParam& ebc_param,
    const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list)
    : resource_manager_(resource_manager),
      core_resource_managers_(core_resource_managers),
      batch_size_(batch_size),
      batch_size_per_gpu_(batch_size / resource_manager->get_global_gpu_count()),
      scalar_type_(scalar_type),
      ebc_param_(ebc_param),
      emb_table_param_list_(emb_table_param_list),
      num_local_gpus_(resource_manager->get_local_gpu_count()),
      num_global_gpus_(resource_manager->get_global_gpu_count()),
      num_features_(ebc_param.num_lookup) {
  resident_feature_tables_ = ebc_param.shard_matrix;

  // construct lookup mappings
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    const embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];
    feature_pooling_factors_.push_back(lookup_param.max_hotness);
    feature_id_to_table_id_map_[lookup_id] = lookup_param.table_id;
    for (size_t group_id = 0; group_id < ebc_param.grouped_emb_params.size(); ++group_id) {
      auto group_table_ids = ebc_param.grouped_emb_params[group_id].table_ids;
      if (std::find(group_table_ids.begin(), group_table_ids.end(), lookup_param.table_id) !=
          group_table_ids.end()) {
        feature_id_to_group_id_map_[lookup_id] = group_id;
      }
    }
  }

  init_comm_data();
  init_key_filter();
  init_batch_major_fullbatch_input_preprocessor();
  init_indices_converter();
  init_filtered_all_to_all();
}

void DataDistributor::init_comm_data() {
  // Get number of features in each group
  size_t num_features = 0;
  size_t num_buckets = 0;
  for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
    const auto& lookup_param = ebc_param_.lookup_params[lookup_id];
    num_features += lookup_param.max_hotness;
    num_buckets += 1;
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

    size_t num_bucket_ranges = num_buckets * ebc_param_.universal_batch_size + 1;
    comm_data.bucket_range = core23::Tensor(
        params.shape({static_cast<int64_t>(num_bucket_ranges)}).data_type(ebc_param_.offset_type));

    std::vector<int> hotness_bucket_range(1, 0);
    std::copy(feature_pooling_factors_.begin(), feature_pooling_factors_.end(),
              back_inserter(hotness_bucket_range));
    std::inclusive_scan(hotness_bucket_range.begin() + 1, hotness_bucket_range.end(),
                        hotness_bucket_range.begin() + 1);

    core23::copy_sync(comm_data.hotness_bucket_range, hotness_bucket_range);

    gpu_comm_data_.emplace_back(comm_data);
  }
}

DataDistributor::MPTempStorage::MPTempStorage(std::shared_ptr<core::CoreResourceManager> core,
                                              int batch_size, int sample_max_nnz,
                                              int max_local_features, int max_local_buckets,
                                              core23::DataType key_type,
                                              core23::DataType offset_type) {
  CudaDeviceContext ctx(core->get_device_id());

  int batch_size_per_dev = batch_size / core->get_global_gpu_count();

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
      // ATTENTION: cub radix sort requires NumItemT to be consistent
      cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                      (KeyType*)nullptr, (KeyType*)nullptr,
                                      static_cast<int64_t>(batch_size_per_dev * sample_max_nnz));
    });
    this->temp_sort_storage = core23::Tensor(
        params.shape({static_cast<int64_t>(temp_bytes)}).data_type(core23::ScalarType::Char));
  }
  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION_CORE23(offset_type.type(), BucketRangeType, [&] {
      cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (BucketRangeType*)nullptr,
                                    (BucketRangeType*)nullptr, batch_size * max_local_buckets + 1);
    });
    this->temp_scan_storage = core23::Tensor(
        params.shape({static_cast<int64_t>(temp_bytes)}).data_type(core23::ScalarType::Char));
  }

  this->k_per_b_gpu_major = core23::Tensor(
      params.shape({static_cast<int64_t>(batch_size * max_local_buckets)}).data_type(offset_type));
  this->k_per_b_feat_major = core23::Tensor(
      params.shape({static_cast<int64_t>(batch_size * max_local_buckets)}).data_type(offset_type));
  this->k_per_g = core23::Tensor(
      params.shape({static_cast<int64_t>(core->get_global_gpu_count())}).data_type(offset_type));
  this->bucket_range_gpu_major =
      core23::Tensor(params.shape({static_cast<int64_t>(batch_size * max_local_buckets + 1)})
                         .data_type(offset_type));
  this->sorted_local_keys =
      core23::Tensor(params.shape({static_cast<int64_t>(batch_size_per_dev * sample_max_nnz)})
                         .data_type(key_type));
  this->sorted_local_labels =
      core23::Tensor(params.shape({static_cast<int64_t>(batch_size_per_dev * sample_max_nnz)})
                         .data_type(core23::ScalarType::UInt32));
  this->keys = core23::Tensor(
      params.shape({static_cast<int64_t>(batch_size * max_local_features)}).data_type(key_type));

  DISPATCH_INTEGRAL_FUNCTION_CORE23(offset_type.type(), offset_t, [&] {
    HCTR_LIB_THROW(cudaMallocHost((void**)&this->h_send_k_per_g,
                                  core->get_global_gpu_count() * sizeof(offset_t)));
    HCTR_LIB_THROW(cudaMallocHost((void**)&this->h_recv_k_per_g,
                                  core->get_global_gpu_count() * sizeof(offset_t)));
  });
}

void DataDistributor::init_filtered_all_to_all() {
  // --- allocate operators ---
  for (size_t group_id = 0; group_id < ebc_param_.grouped_emb_params.size(); group_id++) {
    if (ebc_param_.grouped_emb_params[group_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      std::vector<mp::LabelAndCountKeysOperator::Result> label_and_count_outputs;
      std::vector<mp::LabelAndCountKeysOperator> label_and_count_operators;
      std::vector<mp::CountKeysOperator> count_operators;
      std::vector<mp::TransposeBucketsOperator> transpose_operators;
      std::vector<mp::SwizzleKeysOperator> swizzle_operators;

      for (size_t i = 0; i < num_local_gpus_; ++i) {
        CudaDeviceContext context(core_resource_managers_[i]->get_device_id());

        label_and_count_outputs.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
        label_and_count_operators.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
        count_operators.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
        transpose_operators.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
        swizzle_operators.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
      }

      label_and_count_keys_outputs_.push_back(std::move(label_and_count_outputs));
      label_and_count_keys_operators_.push_back(std::move(label_and_count_operators));
      count_keys_operators_.push_back(std::move(count_operators));
      transpose_buckets_operators_.push_back(std::move(transpose_operators));
      swizzle_keys_operators_.push_back(std::move(swizzle_operators));
    } else if (ebc_param_.grouped_emb_params[group_id].table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      std::vector<dp::ConcatKeysAndBucketRangeOperator> concat_operators;
      for (size_t i = 0; i < num_local_gpus_; ++i) {
        CudaDeviceContext context(core_resource_managers_[i]->get_device_id());
        concat_operators.emplace_back(core_resource_managers_[i], ebc_param_, group_id);
      }
      concat_keys_and_bucket_range_operators_.push_back(std::move(concat_operators));
    }
  }

  // -- non-group specific operators
  for (size_t gpu_id = 0; gpu_id < num_local_gpus_; ++gpu_id) {
    compute_dp_bucket_range_operators_.emplace_back(core_resource_managers_[gpu_id], ebc_param_);
  }

  // ---- allocate temp storage ----
  sample_max_nnz_ = 0;
  for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
    sample_max_nnz_ += ebc_param_.lookup_params[lookup_id].max_hotness;
  }

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      std::vector<MPTempStorage> group_temp_storage;
      for (size_t gpu_id = 0; gpu_id < num_local_gpus_; ++gpu_id) {
        auto core = core_resource_managers_[gpu_id];
        int max_local_features = 0;
        int max_local_buckets = 0;
        for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
          if (ebc_param_.has_table_shard(core->get_global_gpu_id(), grouped_id, lookup_id)) {
            max_local_features += ebc_param_.lookup_params[lookup_id].max_hotness;
            max_local_buckets++;
          }
        }
        group_temp_storage.emplace_back(core, ebc_param_.universal_batch_size, sample_max_nnz_,
                                        max_local_features, max_local_buckets, ebc_param_.key_type,
                                        ebc_param_.offset_type);
      }
      temp_storage_.push_back(std::move(group_temp_storage));
    }
  }

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
}

DataDistributor::KeyFilterInitParams::KeyFilterInitParams(
    const std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id)
    : num_lookup(ebc_param.num_lookup),
      global_gpu_id(core_resource_manager->get_global_gpu_id()),
      total_gpu_count(core_resource_manager->get_global_gpu_count()) {
  CudaDeviceContext context(core_resource_manager->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core_resource_manager->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  const auto& lookup_params = ebc_param.lookup_params;
  const auto& group_params = ebc_param.grouped_emb_params[grouped_id];

  size_t num_gpus = core_resource_manager->get_global_gpu_count();
  int gpu_id = core_resource_manager->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  std::vector<int> h_local_lookup_ids;
  std::vector<int> h_local_shard_ids;
  std::vector<int> h_local_num_shards;
  std::vector<int> h_hotness;
  std::vector<int> h_local_hotness;
  for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    h_hotness.push_back(lookup_params[lookup_id].max_hotness);

    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }
    if (ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::DataParallel) {
      HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                      "dp table must be shared on all gpus");
    }

    if (ebc_param.shard_matrix[gpu_id][table_id] == 0) {
      continue;
    }
    h_local_lookup_ids.push_back(lookup_id);
    h_local_hotness.push_back(lookup_params[lookup_id].max_hotness);

    if (ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      std::vector<int> shard_gpus;
      for (size_t ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
        if (ebc_param.shard_matrix[ggpu_id][table_id] == 1) {
          shard_gpus.push_back(ggpu_id);
        }
      }
      auto find_shard_id_iter = std::find(shard_gpus.begin(), shard_gpus.end(), gpu_id);
      HCTR_CHECK_HINT(find_shard_id_iter != shard_gpus.end(),
                      "ModelParallelEmbeddingMeta does not find shard id");
      int shard_id = std::distance(shard_gpus.begin(), find_shard_id_iter);
      h_local_shard_ids.push_back(shard_id);
      h_local_num_shards.push_back(static_cast<int>(shard_gpus.size()));
    }
  }

  num_local_lookup = static_cast<int>(h_local_lookup_ids.size());

  num_hotness = std::accumulate(h_hotness.begin(), h_hotness.end(), 0);

  num_local_hotness = std::accumulate(h_local_hotness.begin(), h_local_hotness.end(), 0);

  if (num_local_lookup) {
    d_local_lookup_ids =
        core23::Tensor(params.shape({static_cast<int64_t>(h_local_lookup_ids.size())})
                           .data_type(core23::ScalarType::Int32));
    core23::copy_sync(d_local_lookup_ids, h_local_lookup_ids);

    if (group_params.table_placement_strategy == embedding::TablePlacementStrategy::ModelParallel) {
      d_local_shard_ids =
          core23::Tensor(params.shape({static_cast<int64_t>(h_local_shard_ids.size())})
                             .data_type(core23::ScalarType::Int32));
      core23::copy_sync(d_local_shard_ids, h_local_shard_ids);
      d_local_num_shards =
          core23::Tensor(params.shape({static_cast<int64_t>(h_local_num_shards.size())})
                             .data_type(core23::ScalarType::Int32));
      core23::copy_sync(d_local_num_shards, h_local_num_shards);
    }
  }
}

void DataDistributor::init_key_filter() {
  size_t num_local_gpus = resource_manager_->get_local_gpu_count();
  for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    std::vector<KeyFilterInitParams> init_params_for_current_gpu;
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
      init_params_for_current_gpu.emplace_back(core_resource_managers_[local_gpu_id], ebc_param_,
                                               grouped_id);
    }
    key_filters_init_params_.push_back(init_params_for_current_gpu);
  }
  for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    std::vector<KeyFilter> key_filters_for_current_gpu;
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
      auto& grouped_emb_param = ebc_param_.grouped_emb_params[grouped_id];
      auto& init_param = key_filters_init_params_[local_gpu_id][grouped_id];
      KeyFilter key_filter;
      if (grouped_emb_param.table_placement_strategy ==
          embedding::TablePlacementStrategy::ModelParallel) {
        key_filter.mp_key_selector =
            embedding::MPKeySelector{init_param.num_lookup,         init_param.d_local_lookup_ids,
                                     init_param.num_local_lookup,   init_param.d_local_shard_ids,
                                     init_param.d_local_num_shards, init_param.num_hotness,
                                     init_param.num_local_hotness};
        key_filter.mp_index_calculation.init(core_resource_managers_[local_gpu_id],
                                             key_filter.mp_key_selector,
                                             ebc_param_.universal_batch_size);
      } else if (grouped_emb_param.table_placement_strategy ==
                 embedding::TablePlacementStrategy::DataParallel) {
        key_filter.dp_key_selector = embedding::DPKeySelector{
            init_param.num_lookup,       init_param.d_local_lookup_ids, init_param.num_local_lookup,
            init_param.global_gpu_id,    init_param.total_gpu_count,    init_param.num_hotness,
            init_param.num_local_hotness};
        key_filter.dp_index_calculation.init(core_resource_managers_[local_gpu_id],
                                             key_filter.dp_key_selector,
                                             ebc_param_.universal_batch_size);
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "not supported table placement strategy.");
      }
      key_filters_for_current_gpu.push_back(key_filter);
    }
    key_filters_.push_back(key_filters_for_current_gpu);
  }
}

void DataDistributor::init_batch_major_fullbatch_input_preprocessor() {
  if (ebc_param_.input_layout_ == embedding::EmbeddingLayout::BatchMajor) {
    preprocess_inputs_.clear();

    size_t num_local_gpus = resource_manager_->get_local_gpu_count();
    for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      CudaDeviceContext context(core_resource_managers_[local_gpu_id]->get_device_id());

      preprocess_inputs_.push_back(std::make_unique<embedding::PreprocessInput>(
          core_resource_managers_[local_gpu_id], ebc_param_));
    }
  }
}

void DataDistributor::init_indices_converter() {
  if (ebc_param_.keys_preprocess_strategy_ != embedding::KeysPreprocessStrategy::AddOffset) return;
  int num_gpus = resource_manager_->get_local_gpu_count();
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    CudaDeviceContext context(core_resource_managers_[gpu_id]->get_device_id());
    int ggpu_id = core_resource_managers_[gpu_id]->get_global_gpu_id();
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); grouped_id++) {
      indices_converters_.emplace_back(core_resource_managers_[gpu_id], emb_table_param_list_,
                                       ebc_param_, grouped_id);

      std::vector<int> h_local_lookup_id_list;
      std::vector<int> h_local_table_id_list;

      for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
        if (!ebc_param_.has_table_shard(ggpu_id, grouped_id, lookup_id)) continue;
        int table_id = ebc_param_.lookup_params[lookup_id].table_id;

        h_local_lookup_id_list.push_back(lookup_id);
        h_local_table_id_list.push_back(table_id);
      }
      compress_offsets_.push_back(embedding::CompressOffset(core_resource_managers_[gpu_id],
                                                            h_local_lookup_id_list.size() + 1,
                                                            ebc_param_.offset_type));

      core23::Device device(core23::DeviceType::GPU,
                            core_resource_managers_[gpu_id]->get_device_id());
      core23::TensorParams params = core23::TensorParams().device(device);

      core23::Tensor d_local_table_id_list =
          core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list.size())})
                             .data_type(core23::ScalarType::Int32));
      core23::copy_sync(d_local_table_id_list, h_local_table_id_list);
      d_local_table_id_lists_.push_back(d_local_table_id_list);
    }
  }
}

void DataDistributor::all2all_keys_per_bucket(int mp_group_i, int gpu_id) {
  auto core = core_resource_managers_[gpu_id];
  auto stream = core->get_local_gpu()->get_stream();

  const auto& per_gpu_lookup_range =
      label_and_count_keys_operators_[mp_group_i][gpu_id].h_per_gpu_lookup_range;
  auto send_tensor = label_and_count_keys_outputs_[mp_group_i][gpu_id].keys_per_bucket;
  auto recv_tensor = temp_storage_[mp_group_i][gpu_id].k_per_b_gpu_major;
  auto nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensor.data_type().type());

  size_t recv_num_buckets =
      per_gpu_lookup_range[(core->get_global_gpu_id() + 1) * ebc_param_.num_lookup] -
      per_gpu_lookup_range[core->get_global_gpu_id() * ebc_param_.num_lookup];

  size_t recv_offset = 0;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_tensor.data_type().type(), BucketRangeType, [&] {
    ncclGroupStart();
    for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
      size_t start_range = per_gpu_lookup_range[peer * ebc_param_.num_lookup];
      size_t send_num_buckets =
          per_gpu_lookup_range[(peer + 1) * ebc_param_.num_lookup] - start_range;

      HCTR_LIB_THROW(ncclSend(send_tensor.data<BucketRangeType>() + start_range, send_num_buckets,
                              nccl_type, peer, core->get_nccl(), stream));
      HCTR_LIB_THROW(ncclRecv(recv_tensor.data<BucketRangeType>() + recv_offset, recv_num_buckets,
                              nccl_type, peer, core->get_nccl(), stream));

      recv_offset += recv_num_buckets;
    }
    ncclGroupEnd();
  });
}

void DataDistributor::all2all_keys(int mp_group_i, int gpu_id, size_t& received_num_keys) {
  auto core = core_resource_managers_[gpu_id];
  auto stream = core->get_local_gpu()->get_stream();

  auto send_tensor = temp_storage_[mp_group_i][gpu_id].sorted_local_keys;
  auto recv_tensor = temp_storage_[mp_group_i][gpu_id].keys;

  auto send_k_per_g = label_and_count_keys_outputs_[mp_group_i][gpu_id].keys_per_gpu;
  auto recv_k_per_g = temp_storage_[mp_group_i][gpu_id].k_per_g;

  auto nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensor.data_type().type());

  // Prefetch counts to host
  HCTR_LIB_THROW(cudaMemcpyAsync(temp_storage_[mp_group_i][gpu_id].h_send_k_per_g,
                                 send_k_per_g.data(), send_k_per_g.num_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaMemcpyAsync(temp_storage_[mp_group_i][gpu_id].h_recv_k_per_g,
                                 recv_k_per_g.data(), recv_k_per_g.num_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  size_t send_offset = 0;
  size_t recv_offset = 0;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_tensor.data_type().type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(send_k_per_g.data_type().type(), BucketRangeType, [&] {
      ncclGroupStart();
      for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
        auto send_num_keys =
            static_cast<BucketRangeType*>(temp_storage_[mp_group_i][gpu_id].h_send_k_per_g)[peer];
        auto recv_num_keys =
            static_cast<BucketRangeType*>(temp_storage_[mp_group_i][gpu_id].h_recv_k_per_g)[peer];
        //        printf("GPU (%d) -> (%d) send_num_keys: %d, recv_num_keys: %d\n",
        //        core->get_global_gpu_id(),
        //               (int)peer, (int)send_num_keys, (int)recv_num_keys);
        if (send_num_keys > 0) {
          HCTR_LIB_THROW(ncclSend(send_tensor.data<KeyType>() + send_offset, send_num_keys,
                                  nccl_type, peer, core->get_nccl(), stream));
        }
        if (recv_num_keys > 0) {
          HCTR_LIB_THROW(ncclRecv(recv_tensor.data<KeyType>() + recv_offset, recv_num_keys,
                                  nccl_type, peer, core->get_nccl(), stream));
        }
        send_offset += send_num_keys;
        recv_offset += recv_num_keys;
      }
      ncclGroupEnd();
    });
  });

  received_num_keys = recv_offset;
}

void DataDistributor::key_filtered_distribute(int gpu_id,
                                              const std::vector<core23::Tensor>& dp_keys,
                                              const std::vector<core23::Tensor>& dp_bucket_range,
                                              DataDistributor::Result& output, int batch_size) {
  auto core = core_resource_managers_[gpu_id];
  CudaDeviceContext ctx(core->get_device_id());
  cudaStream_t stream = core->get_local_gpu()->get_stream();

  size_t mp_group_i = 0;
  size_t dp_group_i = 0;

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); grouped_id++) {
    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      // --- Label keys for sort, and count keys per bucket & GPU ---
      {
        auto compute = label_and_count_keys_operators_[mp_group_i][gpu_id];
        auto& result = label_and_count_keys_outputs_[mp_group_i][gpu_id];

        compute(dp_keys, dp_bucket_range, result, stream);
      }

      // --- Inter-node traffic ---
      all2all_keys_per_bucket(mp_group_i, gpu_id);

      // --- count keys per gpu, so we know how many keys to receive from nccl
      // TODO: compare with cub::DeviceSegmentedReduce::Sum
      {
        auto compute = count_keys_operators_[mp_group_i][gpu_id];
        auto k_per_b_gpu_major = temp_storage_[mp_group_i][gpu_id].k_per_b_gpu_major;
        auto k_per_g = temp_storage_[mp_group_i][gpu_id].k_per_g;

        compute(k_per_b_gpu_major, k_per_g, stream);
      }

      // --- sort keys by global gpu id ---
      DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.key_type.type(), KeyType, [&] {
        size_t temp_bytes = temp_storage_[mp_group_i][gpu_id].temp_sort_storage.num_bytes();
        void* temp_ptr = temp_storage_[mp_group_i][gpu_id].temp_sort_storage.data();
        auto keys_in = label_and_count_keys_outputs_[mp_group_i][gpu_id].flat_keys.data<KeyType>();
        auto keys_out = temp_storage_[mp_group_i][gpu_id].sorted_local_keys.data<KeyType>();
        auto labels_in =
            label_and_count_keys_outputs_[mp_group_i][gpu_id].local_labels.data<uint32_t>();
        auto labels_out = temp_storage_[mp_group_i][gpu_id].sorted_local_labels.data<uint32_t>();

        // ATTENTION: cub radix sort requires NumItemT to be consistent
        int sort_end_bit = (int)log2(num_global_gpus_) + 1;

        cub::DeviceRadixSort::SortPairs(
            temp_ptr, temp_bytes, labels_in, labels_out, keys_in, keys_out,
            static_cast<int64_t>(batch_size_per_gpu_ * sample_max_nnz_), 0, sort_end_bit, stream);
      });

      // --- Inter-node traffic ---
      all2all_keys(mp_group_i, gpu_id, output[grouped_id].h_num_keys);

      {
        auto compute = transpose_buckets_operators_[mp_group_i][gpu_id];
        compute(temp_storage_[mp_group_i][gpu_id].k_per_b_gpu_major,
                temp_storage_[mp_group_i][gpu_id].k_per_b_feat_major, stream);
      }

      // --- computes bucket ranges received from nccl
      DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
        size_t temp_bytes = temp_storage_[mp_group_i][gpu_id].temp_scan_storage.num_bytes();
        void* temp_ptr = temp_storage_[mp_group_i][gpu_id].temp_scan_storage.data();
        auto k_per_b_gpu_major = temp_storage_[mp_group_i][gpu_id].k_per_b_gpu_major;
        auto bucket_range_gpu_major = temp_storage_[mp_group_i][gpu_id].bucket_range_gpu_major;

        // computes in-place!
        HCTR_LIB_THROW(cudaMemsetAsync(bucket_range_gpu_major.data<BucketRangeType>(), 0,
                                       sizeof(BucketRangeType), stream));
        cub::DeviceScan::InclusiveSum(temp_ptr, temp_bytes,
                                      k_per_b_gpu_major.data<BucketRangeType>(),
                                      bucket_range_gpu_major.data<BucketRangeType>() + 1,
                                      k_per_b_gpu_major.num_elements(), stream);
      });

      // --- computes output bucket range
      DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
        size_t temp_bytes = temp_storage_[mp_group_i][gpu_id].temp_scan_storage.num_bytes();
        void* temp_ptr = temp_storage_[mp_group_i][gpu_id].temp_scan_storage.data();
        auto k_per_b_feat_major = temp_storage_[mp_group_i][gpu_id].k_per_b_feat_major;

        HCTR_LIB_THROW(cudaMemsetAsync(output[grouped_id].bucket_range.data<BucketRangeType>(), 0,
                                       sizeof(BucketRangeType), stream));
        cub::DeviceScan::InclusiveSum(temp_ptr, temp_bytes,
                                      k_per_b_feat_major.data<BucketRangeType>(),
                                      output[grouped_id].bucket_range.data<BucketRangeType>() + 1,
                                      k_per_b_feat_major.num_elements(), stream);
      });

      // --- swizzle keys from gpu-major to feature-major
      {
        auto compute = swizzle_keys_operators_[mp_group_i][gpu_id];
        auto keys_in = temp_storage_[mp_group_i][gpu_id].keys;
        auto bucket_range_in = temp_storage_[mp_group_i][gpu_id].bucket_range_gpu_major;
        auto bucket_range_out = output[grouped_id].bucket_range;

        compute(bucket_range_in, bucket_range_out, keys_in, output[grouped_id].keys, stream);
      }

      mp_group_i++;
    } else if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      // --- copy DP keys and compute DP bucket range
      concat_keys_and_bucket_range_operators_[dp_group_i][gpu_id](
          dp_keys, dp_bucket_range, output[grouped_id].keys, output[grouped_id].bucket_range,
          stream);

      DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
        BucketRangeType num_keys = 0;
        int num_buckets = output[grouped_id].bucket_range.num_elements();
        HCTR_LIB_THROW(cudaMemcpyAsync(
            &num_keys, output[grouped_id].bucket_range.data<BucketRangeType>() + num_buckets - 1,
            sizeof(BucketRangeType), cudaMemcpyDeviceToHost, stream));
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
        output[grouped_id].h_num_keys = num_keys;
      });

      dp_group_i++;

    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unsupported table placement strategy.");
    }
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

  // compute new full batch bucket range (to be deprecated)
  // compute dp bucket ranges (to be moved to data reader)
  if (bucket_ranges_outdated) {
    compute_fixed_bucket_ranges(gpu_comm_data_[gpu_id].hotness_bucket_range, batch_size,
                                batch_size_, gpu_comm_data_[gpu_id].bucket_range, stream);

    compute_dp_bucket_range_operators_[gpu_id](fixed_dp_bucket_range_[gpu_id],
                                               output[0].num_keys_per_bucket, batch_size, stream);

    // Instead of recomputing for each group, copy computed result
    for (size_t grouped_id = 1; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
      HCTR_LIB_THROW(cudaMemcpyAsync(
          output[grouped_id].num_keys_per_bucket.data(), output[0].num_keys_per_bucket.data(),
          output[0].num_keys_per_bucket.num_bytes(), cudaMemcpyDeviceToDevice, stream));
    }
  }

  key_filtered_distribute(gpu_id, dp_keys, fixed_dp_bucket_range_[gpu_id], output, batch_size);

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
    int batch_size_per_gpu = batch_size_;
    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::DataParallel) {
      batch_size_per_gpu /= num_global_gpus_;
    }

    int num_groups = ebc_param_.grouped_emb_params.size();
    if (!indices_converters_.empty()) {
      core23::Tensor num_keys_per_lookup_offset;
      compress_offsets_[gpu_id * num_groups + grouped_id].compute(
          output[grouped_id].bucket_range, batch_size_per_gpu, &num_keys_per_lookup_offset);

      indices_converters_[gpu_id * num_groups + grouped_id].convert(
          output[grouped_id].keys, output[grouped_id].h_num_keys, num_keys_per_lookup_offset,
          num_keys_per_lookup_offset.num_elements() - 1,
          d_local_table_id_lists_[gpu_id * num_groups + grouped_id]);
    }
  }
}

namespace {
template <typename BucketRangeType>
__global__ void cal_num_key_per_bucket_from_fullbatch_bucket_range_kernel(
    const BucketRangeType* __restrict__ fullbatch_bucket_range_ptr,
    BucketRangeType* num_key_per_bucket, int gpu_id, int num_lookup, int batch_size_per_gpu) {
  CUDA_1D_KERNEL_LOOP(i, batch_size_per_gpu * num_lookup) {
    int idx = i + gpu_id * batch_size_per_gpu * num_lookup;
    num_key_per_bucket[i] = fullbatch_bucket_range_ptr[idx + 1] - fullbatch_bucket_range_ptr[idx];
  }
}
}  // namespace

void DataDistributor::distribute(int gpu_id, const core23::Tensor& fullbatch_keys,
                                 const core23::Tensor& fullbatch_bucket_range,
                                 DataDistributor::Result& output, int batch_size) {
  HCTR_ASSERT(ebc_param_.input_layout_ == embedding::EmbeddingLayout::BatchMajor);
  CudaDeviceContext context(core_resource_managers_[gpu_id]->get_device_id());

  auto& preprocess_input = preprocess_inputs_[gpu_id];
  core23::Tensor feature_major_key, feature_major_bucket_range;
  preprocess_input->compute(fullbatch_keys, fullbatch_bucket_range, &feature_major_key,
                            &feature_major_bucket_range, batch_size);

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
    int batch_size_per_gpu = batch_size_;

    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      key_filters_[gpu_id][grouped_id].mp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      batch_size_per_gpu /= num_global_gpus_;
      key_filters_[gpu_id][grouped_id].dp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unsupported table placement strategy.");
    }
    DISPATCH_INTEGRAL_FUNCTION_CORE23(
        fullbatch_bucket_range.data_type().type(), bucket_range_type, [&] {
          int block_size = 256;
          int num_sms = core_resource_managers_[gpu_id]->kernel_params_.num_sms;
          int max_thread_per_block =
              core_resource_managers_[gpu_id]->get_kernel_param().max_thread_per_block;
          int grid_size = (num_sms * max_thread_per_block - 1) / block_size + 1;
          auto stream = core_resource_managers_[gpu_id]->get_local_gpu()->get_stream();
          cal_num_key_per_bucket_from_fullbatch_bucket_range_kernel<<<grid_size, block_size, 0,
                                                                      stream>>>(
              fullbatch_bucket_range.data<bucket_range_type>(),
              output[grouped_id].num_keys_per_bucket.data<bucket_range_type>(),
              core_resource_managers_[gpu_id]->get_global_gpu_id(), ebc_param_.num_lookup,
              batch_size_ / num_global_gpus_);
        });

    int num_groups = ebc_param_.grouped_emb_params.size();
    if (!indices_converters_.empty()) {
      core23::Tensor num_keys_per_lookup_offset;
      compress_offsets_[gpu_id * num_groups + grouped_id].compute(
          output[grouped_id].bucket_range, batch_size_per_gpu, &num_keys_per_lookup_offset);

      indices_converters_[gpu_id * num_groups + grouped_id].convert(
          output[grouped_id].keys, output[grouped_id].h_num_keys, num_keys_per_lookup_offset,
          num_keys_per_lookup_offset.num_elements() - 1,
          d_local_table_id_lists_[gpu_id * num_groups + grouped_id]);
    }
  }
}

DataDistributor::Result allocate_output_for_data_distributor(
    std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param) {
  CudaDeviceContext context(core_resource_manager->get_device_id());
  int num_global_gpus = core_resource_manager->get_global_gpu_count();
  int batch_size_per_gpu = ebc_param.universal_batch_size / num_global_gpus;

  DataDistributor::Result output;
  for (size_t group_id = 0; group_id < ebc_param.grouped_emb_params.size(); ++group_id) {
    int batch_size_after_filter = ebc_param.grouped_emb_params[group_id].table_placement_strategy ==
                                          embedding::TablePlacementStrategy::DataParallel
                                      ? ebc_param.universal_batch_size / num_global_gpus
                                      : ebc_param.universal_batch_size;
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

    embedding_input.bucket_range = core23::Tensor(
        params.shape({static_cast<int64_t>(batch_size_after_filter * num_buckets + 1)})
            .data_type(ebc_param.offset_type));

    embedding_input.num_keys = core23::Tensor(
        params.shape({1}).data_type(core23::ScalarType::UInt64).device(core23::DeviceType::CPU));

    embedding_input.num_keys_per_bucket = core23::Tensor(
        params.shape({static_cast<int64_t>(batch_size_per_gpu * ebc_param.num_lookup)})
            .data_type(ebc_param.offset_type));

    output.push_back(embedding_input);
  }
  return output;
}
}  // namespace HugeCTR
