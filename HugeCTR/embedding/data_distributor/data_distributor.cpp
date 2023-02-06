#include "data_distributor.hpp"

#include "core/hctr_impl/hctr_backend.hpp"
#include "gpu_kernels.hpp"

namespace HugeCTR {

DataDistributor::DataDistributor(
    size_t batch_size, core::DataType scalar_type,
    std::shared_ptr<ResourceManager> resource_manager,
    std::vector<std::shared_ptr<core::CoreResourceManager>>& core_resource_managers,
    const embedding::EmbeddingCollectionParam& ebc_param)
    : resource_manager_(resource_manager),
      core_resource_managers_(core_resource_managers),
      batch_size_(batch_size),
      scalar_type_(scalar_type),
      ebc_param_(ebc_param),
      num_local_gpus_(resource_manager->get_local_gpu_count()),
      num_features_(ebc_param.num_lookup) {
  std::vector<std::vector<int>> default_residency(num_local_gpus_,
                                                  std::vector<int>(ebc_param.num_table, 1));
  resident_feature_tables_ = default_residency;  // TODO: ebc_param.shard_matrix

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

  init_nccl_comms();
  init_comm_data();
  init_key_filter();
  init_batch_major_fullbatch_input_preprocessor();
}

void DataDistributor::init_nccl_comms() {
#if defined(ENABLE_MPI)
  printf("MPI ENABLED\n");
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks_);

  ncclUniqueId id;
  if (my_rank_ == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
#else
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  my_rank_ = 0;
  n_ranks_ = 1;
#endif

  comms_.resize(num_local_gpus_);
  ncclGroupStart();
  for (size_t i = 0; i < num_local_gpus_; i++) {
    CudaDeviceContext context(core_resource_managers_[i]->get_device_id());
    ncclCommInitRank(&comms_[i], num_local_gpus_ * n_ranks_, id, my_rank_ * num_local_gpus_ + i);
  }
  ncclGroupEnd();
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

    GpuCommData comm_data;
    comm_data.local_rank = i;
    comm_data.last_batch_size = 0;

    auto buffer_ptr = core::GetBuffer(core_resource_managers_[i]);
    size_t num_keys = num_features * ebc_param_.universal_batch_size;

    comm_data.hotness_bucket_range = buffer_ptr->reserve({num_features_ + 1}, core::DeviceType::GPU,
                                                         core::TensorScalarType::Int32);
    comm_data.features =
        buffer_ptr->reserve({num_keys}, core::DeviceType::GPU, ebc_param_.key_type);

    size_t num_bucket_ranges = num_buckets * ebc_param_.universal_batch_size + 1;
    comm_data.bucket_range =
        buffer_ptr->reserve({num_bucket_ranges}, core::DeviceType::GPU, ebc_param_.offset_type);
    buffer_ptr->allocate();

    std::vector<int> hotness_bucket_range(1, 0);
    std::copy(feature_pooling_factors_.begin(), feature_pooling_factors_.end(),
              back_inserter(hotness_bucket_range));
    std::inclusive_scan(hotness_bucket_range.begin() + 1, hotness_bucket_range.end(),
                        hotness_bucket_range.begin() + 1);

    comm_data.hotness_bucket_range.copy_from(hotness_bucket_range);

    init_fixed_bucket_ranges(comm_data.bucket_range);
    gpu_comm_data_.emplace_back(comm_data);
  }
}

DataDistributor::KeyFilterInitParams::KeyFilterInitParams(
    const std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t grouped_id)
    : num_lookup(ebc_param.num_lookup),
      global_gpu_id(core_resource_manager->get_global_gpu_id()),
      total_gpu_count(core_resource_manager->get_global_gpu_count()) {
  CudaDeviceContext context(core_resource_manager->get_device_id());

  const auto& lookup_params = ebc_param.lookup_params;
  auto buffer_ptr = GetBuffer(core_resource_manager);
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

  d_local_lookup_ids = buffer_ptr->reserve({h_local_lookup_ids.size()}, core::DeviceType::GPU,
                                           TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_lookup_ids.copy_from(h_local_lookup_ids);

  d_local_shard_ids = buffer_ptr->reserve({h_local_shard_ids.size()}, core::DeviceType::GPU,
                                          TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_shard_ids.copy_from(h_local_shard_ids);

  d_local_num_shards = buffer_ptr->reserve({h_local_num_shards.size()}, core::DeviceType::GPU,
                                           TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_num_shards.copy_from(h_local_num_shards);
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

void DataDistributor::init_fixed_bucket_ranges(core::Tensor& output_bucket_ranges) const {
  // feature-major bucket ranges
  DISPATCH_INTEGRAL_FUNCTION(output_bucket_ranges.dtype().type(), BucketRangeType, [&] {
    std::vector<BucketRangeType> bucket_ranges(1, 0);
    for (size_t feat_id = 0; feat_id < num_features_; ++feat_id) {
      bucket_ranges.insert(bucket_ranges.end(), batch_size_, feature_pooling_factors_[feat_id]);
    }
    std::inclusive_scan(bucket_ranges.begin() + 1, bucket_ranges.end(), bucket_ranges.begin() + 1);
    // copy up to GPU
    output_bucket_ranges.copy_from(bucket_ranges);
  });
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

DataDistributor::~DataDistributor() {
  for (size_t i = 0; i < num_local_gpus_; i++) {
    ncclCommDestroy(comms_[i]);
  }
}

void DataDistributor::distribute(int gpu_id, const std::vector<core::Tensor>& dp_keys,
                                 const std::vector<core::Tensor>& dp_bucket_range,
                                 DataDistributor::Result& output, int batch_size) {
  assert(gpu_id < gpu_comm_data_.size());
  assert(batch_size <= batch_size_ && "input batch_size larger than allocated batch_size");

  HugeCTR::CudaDeviceContext ctx(resource_manager_->get_local_gpu(gpu_id)->get_device_id());
  auto stream = resource_manager_->get_local_gpu(gpu_id)->get_stream();

  communicate_data(dp_keys, gpu_id, stream);

  const bool bucket_ranges_outdated = batch_size != gpu_comm_data_[gpu_id].last_batch_size;
  gpu_comm_data_[gpu_id].last_batch_size = batch_size;

  if (bucket_ranges_outdated) {
    compute_fixed_bucket_ranges(gpu_comm_data_[gpu_id].hotness_bucket_range, batch_size,
                                batch_size_, gpu_comm_data_[gpu_id].bucket_range, stream);
  }

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      key_filters_[gpu_id][grouped_id].mp_index_calculation.filter_sparse_input(
          gpu_comm_data_[gpu_id].features, gpu_comm_data_[gpu_id].bucket_range, output[grouped_id],
          batch_size_);
    } else if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      key_filters_[gpu_id][grouped_id].dp_index_calculation.filter_sparse_input(
          gpu_comm_data_[gpu_id].features, gpu_comm_data_[gpu_id].bucket_range, output[grouped_id],
          batch_size_);
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unsupported table placement strategy.");
    }

    if (bucket_ranges_outdated) {
      HCTR_LIB_THROW(cudaMemcpyAsync(output[grouped_id].fullbatch_bucket_range.get(),
                                     gpu_comm_data_[gpu_id].bucket_range.get(),
                                     gpu_comm_data_[gpu_id].bucket_range.nbytes(),
                                     cudaMemcpyDeviceToDevice, stream));
    }
  }
}
//
// void DataDistributor::communicate_data(GpuCommData& comm_data,
//                                       std::vector<core::Tensor> feature_shards, ncclComm_t comm,
//                                       cudaStream_t stream, CommResult& output) {
//  std::vector<size_t> group_recv_offsets(ebc_param_.grouped_emb_params.size(), 0);
//
//  const auto nccl_type = get_nccl_dtype_from_tensor_scalar_type(scalar_type_.type());
//
//  ncclGroupStart();
//  for (size_t feat_id = 0; feat_id < feature_shards.size(); ++feat_id) {
//    const size_t table_id = feature_id_to_table_id_map_[feat_id];
//    const size_t group_id = feature_id_to_group_id_map_[feat_id];
//    auto& output_features = output.features[group_id];
//    auto& send_tensor = feature_shards[feat_id];
//
//    switch (ebc_param_.grouped_emb_params[group_id].table_placement_strategy) {
//      case embedding::TablePlacementStrategy::ModelParallel: {
//        for (size_t peer = 0; peer < num_local_gpus_ * n_ranks_; ++peer) {
//          const bool peer_device_has_feature = resident_feature_tables_[peer][table_id];
//          if (peer_device_has_feature)
//            HCTR_LIB_THROW(ncclSend(send_tensor.get(), send_tensor.get_num_elements(), nccl_type,
//                                    peer, comm, stream));
//
//          const bool this_device_has_feature =
//              resident_feature_tables_[comm_data.local_rank][table_id];
//          if (this_device_has_feature) {
//            HCTR_LIB_THROW(ncclRecv((char*)output_features.get() + group_recv_offsets[group_id],
//                                    send_tensor.get_num_elements(), nccl_type, peer, comm,
//                                    stream));
//            group_recv_offsets[group_id] +=
//                send_tensor.nbytes();  // assumes all ranks have same shard size
//          }
//        }
//        break;
//      }
//      case embedding::TablePlacementStrategy::DataParallel: {
//        // coalesce shards
//        HCTR_LIB_THROW(cudaMemcpyAsync((char*)output_features.get() +
//        group_recv_offsets[group_id],
//                                       send_tensor.get(), send_tensor.nbytes(), cudaMemcpyDefault,
//                                       stream));
//        group_recv_offsets[group_id] += send_tensor.nbytes();
//        break;
//      }
//      default:
//        throw std::runtime_error("Table placement strategy not supported in DataDistributor");
//    }
//  }
//  ncclGroupEnd();
//}

void DataDistributor::communicate_data(std::vector<core::Tensor> feature_shards, int gpu_id,
                                       cudaStream_t stream) {
  size_t recv_offset = 0;

  GpuCommData& comm_data = gpu_comm_data_[gpu_id];
  ncclComm_t comm = comms_[gpu_id];

  const auto nccl_type = get_nccl_dtype_from_tensor_scalar_type(scalar_type_.type());

  ncclGroupStart();
  for (size_t feat_id = 0; feat_id < feature_shards.size(); ++feat_id) {
    const size_t table_id = feature_id_to_table_id_map_[feat_id];
    auto recv_buffer = (char*)comm_data.features.get();
    auto send_buffer = (char*)feature_shards[feat_id].get();
    const size_t feature_num_keys = feature_shards[feat_id].get_num_elements();

    for (size_t peer = 0; peer < num_local_gpus_ * n_ranks_; ++peer) {
      const bool peer_device_has_feature = resident_feature_tables_[peer][table_id];
      if (peer_device_has_feature) {
        HCTR_LIB_THROW(ncclSend(send_buffer, feature_num_keys, nccl_type, peer, comm, stream));
      }

      const bool this_device_has_feature = resident_feature_tables_[comm_data.local_rank][table_id];
      if (this_device_has_feature) {
        HCTR_LIB_THROW(
            ncclRecv(recv_buffer + recv_offset, feature_num_keys, nccl_type, peer, comm, stream));
        recv_offset += feature_num_keys * scalar_type_.itemsize();
      }
    }
  }
  ncclGroupEnd();
}

void DataDistributor::distribute(int gpu_id, const core::Tensor& batch_major_fullbatch_keys,
                                 const core::Tensor& batch_major_fullbatch_bucket_range,
                                 DataDistributor::Result& output, int batch_size) {
  HCTR_ASSERT(ebc_param_.input_layout_ == embedding::EmbeddingLayout::BatchMajor);
  CudaDeviceContext context(core_resource_managers_[gpu_id]->get_device_id());

  auto& preprocess_input = preprocess_inputs_[gpu_id];
  core::Tensor feature_major_key, feature_major_bucket_range;
  preprocess_input->compute(batch_major_fullbatch_keys, batch_major_fullbatch_bucket_range,
                            &feature_major_key, &feature_major_bucket_range, batch_size);

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_emb_params.size(); ++grouped_id) {
    if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      key_filters_[gpu_id][grouped_id].mp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else if (ebc_param_.grouped_emb_params[grouped_id].table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      key_filters_[gpu_id][grouped_id].dp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unsupported table placement strategy.");
    }
    output[grouped_id].fullbatch_bucket_range.copy_from(
        feature_major_bucket_range, core_resource_managers_[gpu_id]->get_local_gpu()->get_stream());
  }
}

DataDistributor::Result allocate_output_for_data_distributor(
    std::shared_ptr<core::CoreResourceManager>& core_resource_manager,
    const embedding::EmbeddingCollectionParam& ebc_param) {
  CudaDeviceContext context(core_resource_manager->get_device_id());
  int num_global_gpus = core_resource_manager->get_global_gpu_count();

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

    auto buffer_ptr = core::GetBuffer(core_resource_manager);
    embedding::EmbeddingInput embedding_input;
    embedding_input.h_num_keys = 0ul;
    embedding_input.keys = buffer_ptr->reserve({batch_size_after_filter * num_features},
                                               core::DeviceType::GPU, ebc_param.key_type);
    embedding_input.bucket_range =
        buffer_ptr->reserve({batch_size_after_filter * num_buckets + 1}, core::DeviceType::GPU,
                            core::TensorScalarType::UInt32);
    embedding_input.num_keys =
        buffer_ptr->reserve({1}, core::DeviceType::CPU, core::TensorScalarType::Size_t);
    embedding_input.fullbatch_bucket_range =
        buffer_ptr->reserve({ebc_param.universal_batch_size * ebc_param.num_lookup + 1},
                            core::DeviceType::GPU, ebc_param.offset_type);
    buffer_ptr->allocate();

    output.push_back(embedding_input);
  }
  return output;
}
}  // namespace HugeCTR