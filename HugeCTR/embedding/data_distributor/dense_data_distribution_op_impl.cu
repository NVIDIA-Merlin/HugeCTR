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
#include <HugeCTR/include/utils.hpp>
#include <embedding/data_distributor/data_compression_operators.cuh>
#include <embedding/data_distributor/data_distribution_op.hpp>
#include <embedding/operators/communication.hpp>

namespace HugeCTR {

int calc_sum_num_features_in_current_group(const embedding::EmbeddingCollectionParam& ebc_param,
                                           size_t group_id) {
  int max_num_features_in_current_group = 0;
  for (int lookup_id : ebc_param.grouped_lookup_params[group_id].lookup_ids) {
    max_num_features_in_current_group += ebc_param.lookup_params[lookup_id].max_hotness;
  }
  return max_num_features_in_current_group;
}

int calc_num_features_in_current_gpu(int global_gpu_id,
                                     const embedding::EmbeddingCollectionParam& ebc_param,
                                     size_t group_id) {
  int num_features_in_current_gpu = 0;
  for (int lookup_id : ebc_param.grouped_lookup_params[group_id].lookup_ids) {
    if (!ebc_param.has_table_shard(global_gpu_id, group_id, lookup_id)) continue;
    num_features_in_current_gpu += ebc_param.lookup_params[lookup_id].max_hotness;
  }
  return num_features_in_current_gpu;
}

int calc_max_num_features_in_current_gpu(int global_gpu_id,
                                         const embedding::EmbeddingCollectionParam& ebc_param,
                                         size_t group_id) {
  std::unordered_map<int, int> table_id_to_max_hotness;
  for (int lookup_id : ebc_param.grouped_lookup_params[group_id].lookup_ids) {
    if (!ebc_param.has_table_shard(global_gpu_id, group_id, lookup_id)) continue;
    int table_id = ebc_param.lookup_params[lookup_id].table_id;
    table_id_to_max_hotness[table_id] += ebc_param.lookup_params[lookup_id].max_hotness;
  }

  int max_num_features_in_current_gpu = 0;
  for (auto& [_, max_num_features_per_table] : table_id_to_max_hotness) {
    max_num_features_in_current_gpu =
        std::max(max_num_features_per_table, max_num_features_in_current_gpu);
  }
  return max_num_features_in_current_gpu;
}

std::vector<int> calc_local_lookup_id_to_global_lookup_ids(
    int global_gpu_id, const embedding::EmbeddingCollectionParam& ebc_param, size_t group_id) {
  std::vector<int> local_lookup_id_to_global_lookup_ids;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(global_gpu_id, group_id, lookup_id)) continue;

    local_lookup_id_to_global_lookup_ids.push_back(lookup_id);
  }
  return local_lookup_id_to_global_lookup_ids;
}

DenseMPDataDistributionOp::DenseMPTempStorage::DenseMPTempStorage(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t group_id) {
  CudaDeviceContext ctx(core->get_device_id());

  int sum_num_features_in_current_group =
      calc_sum_num_features_in_current_group(ebc_param, group_id);
  int global_gpu_id = core->get_global_gpu_id();
  int num_features_in_current_gpu =
      calc_num_features_in_current_gpu(global_gpu_id, ebc_param, group_id);
  int max_num_features_in_current_gpu =
      calc_max_num_features_in_current_gpu(global_gpu_id, ebc_param, group_id);

  embedding::WgradAttr wgrad_attr;
  wgrad_attr.init(core, ebc_param, group_id);
  this->num_table = wgrad_attr.num_table;

  int global_gpu_count = core->get_global_gpu_count();
  int batch_size = ebc_param.universal_batch_size;
  int batch_size_per_gpu = ebc_param.universal_batch_size / global_gpu_count;
  auto key_type = ebc_param.key_type;
  auto offset_type = ebc_param.offset_type;
  int num_lookup = ebc_param.num_lookup;
  auto& grouped_lookup_param = ebc_param.grouped_lookup_params[group_id];

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  this->shard_partitioner_ = std::make_unique<ShardPartitioner>(
      core, ebc_param.lookup_params, ebc_param.shard_matrix, grouped_lookup_param.lookup_ids);

  std::vector<int> local_lookup_id_to_global_lookup_ids =
      calc_local_lookup_id_to_global_lookup_ids(global_gpu_id, ebc_param, group_id);

  this->table_partitioner_ = std::make_unique<TablePartitioner>(
      core, num_lookup, local_lookup_id_to_global_lookup_ids, wgrad_attr);

  this->h_num_network_reverse_idx = core23::Tensor(
      params.shape({1}).data_type(core23::ScalarType::UInt64).device(core23::DeviceType::CPU));
  this->partitioned_data_after_shard_matrix_partition = PartitionedData(
      core, global_gpu_count, batch_size_per_gpu * sum_num_features_in_current_group, key_type,
      offset_type);
  this->d_num_key_gpu_major =
      core23::Tensor(params.shape({global_gpu_count}).data_type(offset_type));

  this->keys_gpu_major =
      core23::Tensor(params.shape({batch_size * num_features_in_current_gpu}).data_type(key_type));
  this->feature_ids_gpu_major =
      core23::Tensor(params.shape({batch_size * num_features_in_current_gpu})
                         .data_type(core23::ScalarType::Int32));

  this->partitioned_data_after_table_id_partition =
      PartitionedData(core, wgrad_attr.num_table, batch_size * max_num_features_in_current_gpu,
                      key_type, offset_type);
}

DenseMPDataDistributionOp::DenseMPDataDistributionOp(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t group_id,
    const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list)
    : core_(core),
      ebc_param_(ebc_param),
      num_global_gpus_(core->get_global_gpu_count()),
      dense_temp_storage_(core, ebc_param_, group_id),
      partition_and_unique_operator_(core, ebc_param_, group_id),
      compress_reverse_idx_range_operator_(core),
      compact_partitioned_data_operator_(core, dense_temp_storage_.num_table),
      do_reduction_(ebc_param.grouped_lookup_params[group_id].embedding_group_type ==
                    embedding::EmbeddingGroupType::DenseModelParallelWithReduction) {
  CudaDeviceContext context(core->get_device_id());

  partition_and_unique_operator_.init_hash_table_for_unique(core, ebc_param_.key_type);

  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::AddOffset) {
    indices_converter_ = std::make_unique<embedding::KeysToIndicesConverter>(
        core, emb_table_param_list, ebc_param_, group_id);
  }
}

void DenseMPDataDistributionOp::distribute(const DataDistributionInput& input,
                                           embedding::EmbeddingInput& output, int batch_size,
                                           cudaStream_t stream) {
  // get the network_dst_bucket_ids
  filter_before_all2all(input, output, batch_size, stream);
  all2all_keys_per_bucket(output, stream);
  all2all_keys(output, stream);
  // get the final unique lookup keys
  filter_after_all2all(output, stream);
  convert_indices(output);
}

void DenseMPDataDistributionOp::filter_before_all2all(const DataDistributionInput& input,
                                                      embedding::EmbeddingInput& output,
                                                      int batch_size, cudaStream_t stream) {
  auto& dense_compression_output = output.dense_compression_input;
  if (!do_reduction_) {
    partition_and_unique_operator_.fill_continuous_bucket_ids(
        input, dense_compression_output.model_parallel_compression_input.network_dst_bucket_ids,
        dense_temp_storage_.h_num_network_reverse_idx, batch_size, stream);
  } else {
    partition_and_unique_operator_.fill_continuous_bucket_ids_for_reduction(
        input, dense_compression_output.model_parallel_compression_input.network_dst_bucket_ids,
        dense_temp_storage_.h_num_network_reverse_idx, batch_size, stream);
  }

  CompressedData compressed_data_after_shard_matrix_partition{
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition,
      dense_compression_output.model_parallel_compression_input.network_reverse_idx};
  partition_and_unique_operator_.partition_and_unique_on_dp_input(
      input, *dense_temp_storage_.shard_partitioner_, compressed_data_after_shard_matrix_partition,
      stream);
}

void DenseMPDataDistributionOp::all2all_keys_per_bucket(embedding::EmbeddingInput& output,
                                                        cudaStream_t stream) {
  auto& send_tensor =
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition.d_num_key_per_partition;
  auto& recv_tensor = dense_temp_storage_.d_num_key_gpu_major;
  auto nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensor.data_type().type());

  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_tensor.data_type().type(), BucketRangeType, [&] {
    ncclGroupStart();
    for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
      HCTR_LIB_THROW(ncclSend(send_tensor.data<BucketRangeType>() + peer, 1, nccl_type, peer,
                              core_->get_nccl(), stream));
      HCTR_LIB_THROW(ncclRecv(recv_tensor.data<BucketRangeType>() + peer, 1, nccl_type, peer,
                              core_->get_nccl(), stream));
    }
    ncclGroupEnd();
  });
}

void DenseMPDataDistributionOp::all2all_keys(embedding::EmbeddingInput& output,
                                             cudaStream_t stream) {
  auto send_keys =
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition.partitioned_keys;
  auto recv_keys = dense_temp_storage_.keys_gpu_major;

  auto send_feature_ids =
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition.feature_ids;
  auto recv_feature_ids = dense_temp_storage_.feature_ids_gpu_major;

  auto d_send_k_per_g =
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition.d_num_key_per_partition;
  auto d_recv_k_per_g = dense_temp_storage_.d_num_key_gpu_major;

  auto h_send_k_per_g =
      output.dense_compression_input.model_parallel_compression_input.h_recv_k_per_gpu;
  auto h_recv_k_per_g =
      output.dense_compression_input.model_parallel_compression_input.h_send_k_per_gpu;

  auto key_nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_keys.data_type().type());
  auto feature_id_nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(core23::ScalarType::Int32);

  // Prefetch counts to host
  core23::copy_async(h_send_k_per_g, d_send_k_per_g, stream);
  core23::copy_async(h_recv_k_per_g, d_recv_k_per_g, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  //  {
  //    std::stringstream ss;
  //
  //    ss << "global gpu_id:" << core_->get_device_id() << std::endl;
  //    ss << "gpu_send_tensor:";
  //    for (size_t i = 0; i < num_global_gpus_; ++i) {
  //      ss << h_send_k_per_g.data<uint32_t>()[i] << " ";
  //    }
  //    ss << std::endl;
  //
  //    ss << "gpu_recv_tensor:";
  //    for (size_t i = 0; i < num_global_gpus_; ++i) {
  //      ss << h_recv_k_per_g.data<uint32_t>()[i] << " ";
  //    }
  //    ss << std::endl;
  //
  //    std::cout << ss.str();
  //  }

  size_t send_offset = 0;
  size_t recv_offset = 0;

  size_t max_num_key_per_partition =
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition.max_num_key_per_partition;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_keys.data_type().type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(d_send_k_per_g.data_type().type(), BucketRangeType, [&] {
      ncclGroupStart();
      for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
        auto send_num_keys = h_send_k_per_g.data<BucketRangeType>()[peer];
        auto recv_num_keys = h_recv_k_per_g.data<BucketRangeType>()[peer];
        //        printf("GPU (%d) -> (%d) send_num_keys: %d, recv_num_keys: %d\n",
        //        core->get_global_gpu_id(),
        //               (int)peer, (int)send_num_keys, (int)recv_num_keys);
        if (send_num_keys > 0) {
          HCTR_LIB_THROW(ncclSend(send_keys.data<KeyType>() + peer * max_num_key_per_partition,
                                  send_num_keys, key_nccl_type, peer, core_->get_nccl(), stream));
          HCTR_LIB_THROW(ncclSend(send_feature_ids.data<int>() + peer * max_num_key_per_partition,
                                  send_num_keys, feature_id_nccl_type, peer, core_->get_nccl(),
                                  stream));
        }
        if (recv_num_keys > 0) {
          HCTR_LIB_THROW(ncclRecv(recv_keys.data<KeyType>() + recv_offset, recv_num_keys,
                                  key_nccl_type, peer, core_->get_nccl(), stream));
          HCTR_LIB_THROW(ncclRecv(recv_feature_ids.data<int>() + recv_offset, recv_num_keys,
                                  feature_id_nccl_type, peer, core_->get_nccl(), stream));
        }
        send_offset += send_num_keys;
        recv_offset += recv_num_keys;
      }
      ncclGroupEnd();
    });
  });

  output.dense_compression_input.model_parallel_compression_input.num_model_reverse_idx =
      recv_offset;
}

void DenseMPDataDistributionOp::filter_after_all2all(embedding::EmbeddingInput& output,
                                                     cudaStream_t stream) {
  auto& dense_compression_output = output.dense_compression_input;

  CompressedData compressed_data_after_table_id_partition{
      dense_temp_storage_.partitioned_data_after_table_id_partition,
      dense_compression_output.model_parallel_compression_input.model_reverse_idx};
  partition_and_unique_operator_.partition_and_unique_by_table_id(
      dense_temp_storage_.keys_gpu_major, dense_temp_storage_.feature_ids_gpu_major,
      output.dense_compression_input.model_parallel_compression_input.num_model_reverse_idx,
      *dense_temp_storage_.table_partitioner_, compressed_data_after_table_id_partition, stream);

  compress_reverse_idx_range_operator_(
      output.dense_compression_input.model_parallel_compression_input.num_model_reverse_idx,
      compressed_data_after_table_id_partition, stream);

  CompressedData compressed_data_after_shard_matrix_partition{
      dense_temp_storage_.partitioned_data_after_shard_matrix_partition,
      dense_compression_output.model_parallel_compression_input.network_reverse_idx};
  dense_compression_output.model_parallel_compression_input.num_network_reverse_idx =
      *(dense_temp_storage_.h_num_network_reverse_idx.data<uint64_t>());
  compress_reverse_idx_range_operator_(
      dense_compression_output.model_parallel_compression_input.num_network_reverse_idx,
      compressed_data_after_shard_matrix_partition, stream);
  CompactedPartitionData continuous_partition_data{
      output.keys, output.num_keys, dense_compression_output.num_keys_per_table_offset};
  compact_partitioned_data_operator_(dense_temp_storage_.partitioned_data_after_table_id_partition,
                                     continuous_partition_data, stream);

  // there is already a sync in compact_partition_data so we dont need sync here
  output.h_num_keys = *(output.num_keys.data<uint64_t>());
}

void DenseMPDataDistributionOp::convert_indices(embedding::EmbeddingInput& output) {
  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::None) return;

  indices_converter_->convert(output.keys, output.h_num_keys,
                              output.dense_compression_input.num_keys_per_table_offset,
                              output.dense_compression_input.table_ids);
}
}  // namespace HugeCTR
