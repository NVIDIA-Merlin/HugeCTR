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

#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <embedding/view.hpp>
#include <utils.cuh>
#include <utils.hpp>

#include "data_compression_operators.hpp"

namespace HugeCTR {

using embedding::bs_upper_bound_sub_one;

template <typename BucketRangeType>
__global__ void cal_range_on_selected_lookup_ids_kernel(
    const BucketRangeType **__restrict__ bucket_range, const int *__restrict__ lookup_ids,
    int num_local_lookup, int num_sample_per_lookup, BucketRangeType *range_on_lookup_ids) {
  BucketRangeType range = 0;
  for (int i = 0; i < num_local_lookup; ++i) {
    range_on_lookup_ids[i] = range;
    range += bucket_range[lookup_ids[i]][num_sample_per_lookup];
  }
}

template <typename KeyType, typename BucketRangeType, typename HashTable>
__global__ void partition_and_unique_kernel(
    const KeyType **keys, const BucketRangeType **bucket_range, const int *lookup_ids,
    const BucketRangeType *range_on_lookup_ids, int num_local_lookup, int num_sample_per_lookup,
    HashTable hash_table, CompressedDataView<KeyType, BucketRangeType> compressed_data) {
  //  extern __shared__ uint64_t smem_bucket_range[];
  //  CUDA_1D_KERNEL_LOOP(i, num_feature * num_sample_per_feature) {
  //    int lookup_id = lookup_ids[i / num_sample_per_feature];
  //    int sample_id = i % num_sample_per_feature;
  //    smem_bucket_range[threadIdx.x] = bucket_range[lookup_id][sample_id];
  //    __syncthreads();
  //
  //    const int num_bucket_per_grid = blockDim.x * gridDim.x;
  //    const int idx_bucket_start_this_block = blockIdx.x * blockDim.x;
  //
  //    const int current_idx_bucket_start_this_block =
  //        (i - i % num_bucket_per_grid) + idx_bucket_start_this_block;
  //
  //    int num_bucket = min(
  //        blockDim.x, num_feature * num_sample_per_feature - current_idx_bucket_start_this_block);
  //
  //    for (int bucket_id = smem_bucket_range[0] + threadIdx.x;
  //         bucket_id < smem_bucket_range[num_bucket]; bucket_id += blockDim.x) {
  //      int idx_bucket = bs_upper_bound_sub_one(smem_bucket_range, num_bucket,
  //      (uint64_t)bucket_id); int feature_id = (idx_bucket + current_idx_bucket_start_this_block)
  //      /
  //                       num_sample_per_feature;  // featrue major
  //      int bid = (idx_bucket + current_idx_bucket_start_this_block) % num_sample_per_feature;
  //      KeyType key = keys[lookup_ids[feature_id]][bid];
  //
  //      uint32_t r_idx_plus_one =
  //          hash_table.find({key, feature_id}, compressed_data.partitioned_data);
  //      compressed_data.reverse_idx[bucket_id] = r_idx_plus_one - 1;
  //    }
  //    __syncthreads();
  //  }

  CUDA_1D_KERNEL_LOOP(i, num_local_lookup * num_sample_per_lookup) {
    BucketRangeType range_start = range_on_lookup_ids[i / num_sample_per_lookup];
    int lookup_id = lookup_ids[i / num_sample_per_lookup];
    int sample_id = i % num_sample_per_lookup;

    BucketRangeType start = bucket_range[lookup_id][sample_id];
    BucketRangeType end = bucket_range[lookup_id][sample_id + 1];

    for (BucketRangeType l = 0; l < end - start; ++l) {
      BucketRangeType local_bucket_id = l + start;

      KeyType key = keys[lookup_id][local_bucket_id];
      uint32_t r_idx_plus_one = hash_table.find({key, lookup_id}, compressed_data.partitioned_data);
      compressed_data.reverse_idx[local_bucket_id + range_start] = r_idx_plus_one - 1;
    }
  }
}

template <typename KeyType, typename BucketRangeType, typename HashTable>
__global__ void partition_and_unique_kernel(
    const KeyType *keys, const int *feature_ids, size_t num_keys, HashTable table,
    CompressedDataView<KeyType, BucketRangeType> compressed_data) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    const KeyType key = keys[i];
    const int feature_id = feature_ids[i];

    uint32_t r_idx_plus_one = table.find({key, feature_id}, compressed_data.partitioned_data);

    compressed_data.reverse_idx[i] = r_idx_plus_one - 1;
  }
}

template <typename BucketRangeType>
__global__ void count_num_bucket_ids(const BucketRangeType **__restrict__ bucket_range,
                                     int num_sample_per_feature, const int *lookup_ids,
                                     size_t num_lookup, uint64_t *num_bucket_ids) {
  CUDA_1D_KERNEL_LOOP(i, num_lookup) {
    int lookup_id = lookup_ids[i];
    uint64_t partial_num_bucket_ids = bucket_range[lookup_id][num_sample_per_feature];
    atomic_add(num_bucket_ids, partial_num_bucket_ids);
  }
}

template <typename BucketRangeType>
__global__ void generate_sequence_kernel(BucketRangeType *bucket_ids, uint64_t *num_bucket_ids) {
  uint64_t num_keys = *num_bucket_ids;
  CUDA_1D_KERNEL_LOOP(i, num_keys) { bucket_ids[i] = i; }
}

template <typename BucketRangeType>
__global__ void compress_reverse_idx_range_kernel(const BucketRangeType *num_key_per_partition,
                                                  int64_t num_partition,
                                                  size_t max_num_key_per_partition,
                                                  BucketRangeType *reverse_idx,
                                                  size_t num_reverse_idx) {
  extern __shared__ uint64_t smem_num_key_per_partition_offset[];
  // FIXME: do scan in collective way
  if (threadIdx.x == 0) {
    BucketRangeType offset = 0;
    for (int i = 0; i < num_partition; ++i) {
      smem_num_key_per_partition_offset[i] = offset;
      offset += num_key_per_partition[i];
    }
  }
  __syncthreads();

  CUDA_1D_KERNEL_LOOP_T(uint64_t, i, num_reverse_idx) {
    BucketRangeType r_idx = reverse_idx[i];
    int partition_id = r_idx / max_num_key_per_partition;
    reverse_idx[i] = r_idx - max_num_key_per_partition * partition_id +
                     smem_num_key_per_partition_offset[partition_id];
  }
}

template <typename KeyType, typename BucketRangeType>
__global__ void compact_keys_kernel(const KeyType *keys, uint64_t num_keys, int num_partition,
                                    size_t max_key_per_partition,
                                    const BucketRangeType *compacted_range, KeyType *compacted_keys,
                                    uint64_t *h_compacted_num_keys) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    int partition_id = i / max_key_per_partition;
    BucketRangeType compacted_range_start = compacted_range[partition_id];
    BucketRangeType compacted_range_end = compacted_range[partition_id + 1];
    BucketRangeType num_key_per_partition = compacted_range_end - compacted_range_start;
    if (i - partition_id * max_key_per_partition < num_key_per_partition) {
      compacted_keys[compacted_range_start + i - partition_id * max_key_per_partition] = keys[i];
    }
  }
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    *h_compacted_num_keys = compacted_range[num_partition];
  }
}

PartitionedData::PartitionedData(std::shared_ptr<core::CoreResourceManager> core,
                                 size_t num_partition, size_t max_num_key_per_partition,
                                 core23::DataType key_type, core23::DataType bucket_range_type)
    : max_num_key_per_partition(max_num_key_per_partition) {
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  this->partitioned_keys =
      core23::Tensor(params.shape({static_cast<int64_t>(max_num_key_per_partition * num_partition)})
                         .data_type(key_type));
  this->feature_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(max_num_key_per_partition * num_partition)})
                         .data_type(core23::ScalarType::Int32));
  this->d_num_key_per_partition = core23::Tensor(
      params.shape({static_cast<int64_t>(num_partition)}).data_type(bucket_range_type));
}

ShardPartitioner::ShardPartitioner(std::shared_ptr<core::CoreResourceManager> core, int num_lookup,
                                   const std::vector<std::vector<int>> &shard_matrix,
                                   const std::vector<int> &lookup_ids) {
  int num_global_gpu_count = core->get_global_gpu_count();

  std::vector<int> h_gpu_ids;
  std::vector<int> h_num_shard_range(num_lookup + 1, 0);
  for (int lookup_id : lookup_ids) {
    int num_shard = 0;
    for (int gpu_id = 0; gpu_id < num_global_gpu_count; ++gpu_id) {
      if (shard_matrix[gpu_id][lookup_id] == 0) continue;
      h_gpu_ids.push_back(gpu_id);
      num_shard += 1;
    }
    h_num_shard_range[lookup_id + 1] = num_shard;
  }
  std::inclusive_scan(h_num_shard_range.begin(), h_num_shard_range.end(),
                      h_num_shard_range.begin());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);
  this->gpu_ids = core23::Tensor(
      params.shape({static_cast<int64_t>(h_gpu_ids.size())}).data_type(core23::ScalarType::Int32));
  this->num_shard_range =
      core23::Tensor(params.shape({static_cast<int64_t>(h_num_shard_range.size())})
                         .data_type(core23::ScalarType::Int32));

  core23::copy_sync(this->gpu_ids, h_gpu_ids);
  core23::copy_sync(this->num_shard_range, h_num_shard_range);
}

TablePartitioner::TablePartitioner(std::shared_ptr<core::CoreResourceManager> core, int num_lookup,
                                   const std::vector<int> &local_lookup_id_to_global_lookup_ids,
                                   const embedding::WgradAttr &wgrad_attr) {
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  std::vector<int> h_lookup_id_to_table_ids(wgrad_attr.lookup_id_to_table_ids.num_elements());
  core23::copy_sync(h_lookup_id_to_table_ids, wgrad_attr.lookup_id_to_table_ids);

  std::vector<int> h_sorted_unique_table_ids(wgrad_attr.sorted_unique_table_ids.num_elements());
  core23::copy_sync(h_sorted_unique_table_ids, wgrad_attr.sorted_unique_table_ids);

  std::vector<int> h_lookup_id_to_local_table_id(num_lookup, -1);
  for (int lookup_id = 0; lookup_id < num_lookup; ++lookup_id) {
    auto iter = std::find(local_lookup_id_to_global_lookup_ids.begin(),
                          local_lookup_id_to_global_lookup_ids.end(), lookup_id);
    if (iter == local_lookup_id_to_global_lookup_ids.end()) continue;

    int local_lookup_id = std::distance(local_lookup_id_to_global_lookup_ids.begin(), iter);
    int table_id = h_lookup_id_to_table_ids[local_lookup_id];

    auto iter_table_id =
        std::find(h_sorted_unique_table_ids.begin(), h_sorted_unique_table_ids.end(), table_id);
    HCTR_CHECK(iter_table_id != h_sorted_unique_table_ids.end());
    int local_table_id = std::distance(h_sorted_unique_table_ids.begin(), iter_table_id);
    h_lookup_id_to_local_table_id[lookup_id] = local_table_id;
  }

  this->lookup_id_to_local_table_id =
      core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_id_to_local_table_id.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->lookup_id_to_local_table_id, h_lookup_id_to_local_table_id);
}

PartitionAndUniqueOperator::PartitionAndUniqueOperator(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam &ebc_param, size_t group_id)
    : num_local_lookup_(ebc_param.grouped_lookup_params[group_id].lookup_ids.size()),
      num_local_features_(0),
      batch_size_(ebc_param.universal_batch_size),
      global_gpu_count_(core->get_global_gpu_count()),
      batch_size_per_gpu_(batch_size_ / global_gpu_count_) {
  CudaDeviceContext ctx(core->get_device_id());

  auto &grouped_lookup_param = ebc_param.grouped_lookup_params[group_id];

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  // 2x for both keys & bucket_range
  this->range_on_lookup_ids =
      core23::Tensor(params.shape({num_local_lookup_}).data_type(ebc_param.offset_type));

  d_lookup_ids_ =
      core23::Tensor(params.shape({num_local_lookup_}).data_type(core23::ScalarType::Int32));

  core23::copy_sync(d_lookup_ids_, grouped_lookup_param.lookup_ids);

  embedding::WgradAttr wgrad_attr;
  wgrad_attr.init(core, ebc_param, group_id);

  for (int lookup_id : grouped_lookup_param.lookup_ids) {
    num_local_features_ += ebc_param.lookup_params[lookup_id].max_hotness;
  }
}

void PartitionAndUniqueOperator::init_hash_table_for_unique(
    std::shared_ptr<core::CoreResourceManager> core, core23::DataType key_type) {
  CudaDeviceContext ctx(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  // TODO: too large
  this->table_capacity_ = batch_size_ * num_local_features_;  // worst case
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    size_t table_size = this->table_capacity_ * sizeof(TableEntry<KeyType>);
    this->hash_table_storage_ = core23::Tensor(
        params.shape({static_cast<int64_t>(table_size)}).data_type(core23::ScalarType::Char));
  });
}

void PartitionAndUniqueOperator::init_hash_table_with_frequent_keys(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::DenseFrequentKeysData &dense_frequent_keys_data) {}

void PartitionAndUniqueOperator::fill_continuous_bucket_ids(const DataDistributionInput &input,
                                                            core23::Tensor &bucket_ids,
                                                            core23::Tensor &num_bucket_ids,
                                                            cudaStream_t stream) {
  HCTR_CHECK(num_bucket_ids.data_type() == core23::ScalarType::UInt64);
  HCTR_LIB_THROW(
      cudaMemsetAsync(num_bucket_ids.data<uint64_t>(), 0, num_bucket_ids.num_bytes(), stream));

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_ids.data_type().type(), BucketRangeType, [&] {
    constexpr int grid_size = 128;
    constexpr int block_size = 256;
    auto bucket_range_ptrs = input.get_dp_bucket_range_pointer_ptr<BucketRangeType>();
    count_num_bucket_ids<<<grid_size, block_size, 0, stream>>>(
        bucket_range_ptrs, batch_size_ / global_gpu_count_, d_lookup_ids_.data<int>(),
        d_lookup_ids_.num_elements(), num_bucket_ids.data<uint64_t>());
  });

  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_ids.data_type().type(), BucketRangeType, [&] {
    constexpr int grid_size = 128;
    constexpr int block_size = 256;
    generate_sequence_kernel<<<grid_size, block_size, 0, stream>>>(
        bucket_ids.data<BucketRangeType>(), num_bucket_ids.data<uint64_t>());
  });
}

template <typename Partitioner>
void PartitionAndUniqueOperator::partition_and_unique_on_dp_input(
    embedding::EmbeddingType embedding_type, const DataDistributionInput &input,
    const Partitioner &partitioner, CompressedData &compressed_data, cudaStream_t stream) {
  auto key_type = compressed_data.partitioned_data.partitioned_keys.data_type();
  auto bucket_range_type = compressed_data.reverse_idx.data_type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range_type.type(), BucketRangeType, [&] {
      auto dp_keys_ptrs = input.get_dp_keys_pointer_ptr<KeyType>();
      auto dp_bucket_range_ptrs = input.get_dp_bucket_range_pointer_ptr<BucketRangeType>();

      cal_range_on_selected_lookup_ids_kernel<<<1, 1, 0, stream>>>(
          dp_bucket_range_ptrs, d_lookup_ids_.data<int>(), num_local_lookup_, batch_size_per_gpu_,
          range_on_lookup_ids.data<BucketRangeType>());

      if (embedding_type == embedding::EmbeddingType::Dense) {
        HCTR_LIB_THROW(cudaMemsetAsync(hash_table_storage_.data(), 0,
                                       hash_table_storage_.num_bytes(), stream));
      }

      HCTR_LIB_THROW(cudaMemsetAsync(
          compressed_data.partitioned_data.d_num_key_per_partition.data(), 0,
          compressed_data.partitioned_data.d_num_key_per_partition.num_bytes(), stream));

      using partitioner_view_type = typename Partitioner::view_type;
      partitioner_view_type partitioner_view = partitioner.view();

      UniqueTableView<KeyType, BucketRangeType, partitioner_view_type> hash_table{
          (TableEntry<KeyType> *)hash_table_storage_.data(), table_capacity_, partitioner_view};
      FrequentTableView<KeyType, BucketRangeType, partitioner_view_type> frequent_hash_table{
          (TableEntry<KeyType> *)frequent_key_hash_table_storage_.data(),
          frequent_key_hash_table_capacity_, partitioner_view};
      InfrequentTableView<KeyType, BucketRangeType, partitioner_view_type> infrequent_hash_table{
          (TableEntry<KeyType> *)frequent_key_hash_table_storage_.data(),
          frequent_key_hash_table_capacity_, partitioner_view};

      CompressedDataView<KeyType, BucketRangeType> compressed_data_view{
          compressed_data.partitioned_data.view<KeyType, BucketRangeType>(),
          compressed_data.reverse_idx.data<BucketRangeType>()};

      constexpr int grid_size = 128;
      constexpr int block_size = 256;
      size_t smem_bytes = block_size * sizeof(BucketRangeType);

      if (embedding_type == embedding::EmbeddingType::Dense) {
        partition_and_unique_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
            dp_keys_ptrs, dp_bucket_range_ptrs, d_lookup_ids_.data<int>(),
            range_on_lookup_ids.data<BucketRangeType>(), num_local_lookup_, batch_size_per_gpu_,
            hash_table, compressed_data_view);
      }
      if (embedding_type == embedding::EmbeddingType::InfrequentDense) {
        partition_and_unique_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
            dp_keys_ptrs, dp_bucket_range_ptrs, d_lookup_ids_.data<int>(),
            range_on_lookup_ids.data<BucketRangeType>(), num_local_lookup_, batch_size_per_gpu_,
            infrequent_hash_table, compressed_data_view);
      }
      if (embedding_type == embedding::EmbeddingType::FrequentDense) {
        partition_and_unique_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
            dp_keys_ptrs, dp_bucket_range_ptrs, d_lookup_ids_.data<int>(),
            range_on_lookup_ids.data<BucketRangeType>(), num_local_lookup_, batch_size_per_gpu_,
            frequent_hash_table, compressed_data_view);
      }
    });
  });
}

template void PartitionAndUniqueOperator::partition_and_unique_on_dp_input(
    embedding::EmbeddingType embedding_type, const DataDistributionInput &input,
    const ShardPartitioner &partitioner, CompressedData &compressed_data, cudaStream_t stream);
template void PartitionAndUniqueOperator::partition_and_unique_on_dp_input(
    embedding::EmbeddingType embedding_type, const DataDistributionInput &input,
    const TablePartitioner &partitioner, CompressedData &compressed_data, cudaStream_t stream);

void PartitionAndUniqueOperator::partition_and_unique_by_table_id(
    const core23::Tensor &keys_gpu_major, const core23::Tensor &feature_ids_gpu_major,
    size_t num_keys, const TablePartitioner &table_partitioner, CompressedData &compressed_data,
    cudaStream_t stream) {
  auto bucket_range_data_type = compressed_data.reverse_idx.data_type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(keys_gpu_major.data_type().type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range_data_type.type(), BucketRangeType, [&] {
      HCTR_LIB_THROW(
          cudaMemsetAsync(hash_table_storage_.data(), 0, hash_table_storage_.num_bytes(), stream));
      HCTR_LIB_THROW(cudaMemsetAsync(
          compressed_data.partitioned_data.d_num_key_per_partition.data(), 0,
          compressed_data.partitioned_data.d_num_key_per_partition.num_bytes(), stream));

      TablePartitionerView partitioner{table_partitioner.lookup_id_to_local_table_id.data<int>()};
      UniqueTableView<KeyType, BucketRangeType, TablePartitionerView> hash_table{
          (TableEntry<KeyType> *)hash_table_storage_.data(), table_capacity_, partitioner};
      CompressedDataView<KeyType, BucketRangeType> compressed_data_view{
          compressed_data.partitioned_data.view<KeyType, BucketRangeType>(),
          compressed_data.reverse_idx.data<BucketRangeType>()};

      constexpr int grid_size = 128;
      constexpr int block_size = 256;

      partition_and_unique_kernel<<<grid_size, block_size, 0, stream>>>(
          keys_gpu_major.data<KeyType>(), feature_ids_gpu_major.data<int>(), num_keys, hash_table,
          compressed_data_view);
    });
  });
}

CompactPartitionDataOperator::CompactPartitionDataOperator(
    std::shared_ptr<core::CoreResourceManager> core, int num_table, core23::DataType offset_type) {
  CudaDeviceContext ctx(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  {
    size_t temp_storage_nbytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_nbytes, (uint64_t *)nullptr,
                                  (uint64_t *)nullptr, num_table);
    this->d_scan_num_key_per_table_temp_storage =
        core23::Tensor(params.shape({static_cast<int64_t>(temp_storage_nbytes)})
                           .data_type(core23::ScalarType::Char));
  }
}

void CompactPartitionDataOperator::operator()(const PartitionedData &partitioned_data,
                                              CompactedPartitionData &compacted_partition_data,
                                              cudaStream_t stream) const {
  auto key_type = partitioned_data.partitioned_keys.data_type();
  auto bucket_range_type = partitioned_data.d_num_key_per_partition.data_type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range_type.type(), BucketRangeType, [&] {
    HCTR_LIB_THROW(cudaMemsetAsync(compacted_partition_data.num_key_per_table.data(), 0,
                                   sizeof(BucketRangeType), stream));

    int num_table = partitioned_data.d_num_key_per_partition.num_elements();
    size_t temp_storage_nbytes = d_scan_num_key_per_table_temp_storage.num_bytes();
    cub::DeviceScan::InclusiveSum(
        d_scan_num_key_per_table_temp_storage.data(), temp_storage_nbytes,
        partitioned_data.d_num_key_per_partition.data<BucketRangeType>(),
        compacted_partition_data.num_key_per_table.data<BucketRangeType>() + 1, num_table, stream);
  });

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range_type.type(), BucketRangeType, [&] {
      constexpr int grid_size = 128;
      constexpr int block_size = 256;
      compact_keys_kernel<<<grid_size, block_size, 0, stream>>>(
          partitioned_data.partitioned_keys.data<KeyType>(),
          partitioned_data.partitioned_keys.num_elements(),
          partitioned_data.d_num_key_per_partition.num_elements(),
          partitioned_data.max_num_key_per_partition,
          compacted_partition_data.num_key_per_table.data<BucketRangeType>(),
          compacted_partition_data.keys.data<KeyType>(),
          compacted_partition_data.h_num_keys.data<uint64_t>());
    });
  });
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

void CompressReverseIdxRangeOperator::operator()(size_t num_bucket_ids,
                                                 CompressedData &compressed_data,
                                                 cudaStream_t stream) const {
  auto bucket_range_type = compressed_data.reverse_idx.data_type();
  HCTR_CHECK(bucket_range_type ==
             compressed_data.partitioned_data.d_num_key_per_partition.data_type());

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range_type.type(), BucketRangeType, [&] {
    constexpr int grid_size = 128;
    constexpr int block_size = 256;
    size_t smem_bytes =
        compressed_data.partitioned_data.d_num_key_per_partition.num_elements() * sizeof(uint64_t);

    compress_reverse_idx_range_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
        compressed_data.partitioned_data.d_num_key_per_partition.data<BucketRangeType>(),
        compressed_data.partitioned_data.d_num_key_per_partition.num_elements(),
        compressed_data.partitioned_data.max_num_key_per_partition,
        compressed_data.reverse_idx.data<BucketRangeType>(), num_bucket_ids);
  });
}
}  // namespace HugeCTR
