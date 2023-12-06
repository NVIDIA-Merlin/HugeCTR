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
#include <cub/cub.cuh>
#include <embedding/data_distributor/data_distribution_op.hpp>
#include <embedding/operators/communication.hpp>

namespace HugeCTR {

SparseDPDataDistributionOp::SparseDPDataDistributionOp(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t group_id,
    const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list)
    : core_(core),
      ebc_param_(ebc_param),
      batch_size_per_gpu_(ebc_param.universal_batch_size / core->get_global_gpu_count()),
      concat_keys_and_bucket_range_operator_(core, ebc_param, group_id) {
  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::AddOffset) {
    indices_converter_ = std::make_unique<embedding::KeysToIndicesConverter>(
        core, emb_table_param_list, ebc_param_, group_id);

    std::vector<int> h_local_lookup_id_list;
    std::vector<int> h_local_table_id_list;

    for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
      if (!ebc_param_.has_table_shard(core->get_global_gpu_id(), group_id, lookup_id)) continue;
      int table_id = ebc_param_.lookup_params[lookup_id].table_id;

      h_local_lookup_id_list.push_back(lookup_id);
      h_local_table_id_list.push_back(table_id);
    }
    compress_offset_ = std::make_unique<embedding::CompressOffset>(
        core, h_local_lookup_id_list.size() + 1, ebc_param_.offset_type);

    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    d_local_table_ids_ =
        core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list.size())})
                           .data_type(core23::ScalarType::Int32));
    core23::copy_sync(d_local_table_ids_, h_local_table_id_list);
  }
}

void SparseDPDataDistributionOp::distribute(const DataDistributionInput& input,
                                            embedding::EmbeddingInput& output, int batch_size,
                                            cudaStream_t stream) {
  // --- copy DP keys and sparse_forward DP bucket range
  concat_keys_and_bucket_range_operator_(input, output.keys, output.bucket_range, stream);

  DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
    BucketRangeType num_keys = 0;
    int num_buckets = output.bucket_range.num_elements();
    HCTR_LIB_THROW(cudaMemcpyAsync(&num_keys,
                                   output.bucket_range.data<BucketRangeType>() + num_buckets - 1,
                                   sizeof(BucketRangeType), cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    output.h_num_keys = num_keys;
  });

  convert_indices(output);
}

void SparseDPDataDistributionOp::convert_indices(embedding::EmbeddingInput& output) {
  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::None) return;
  core23::Tensor num_keys_per_lookup_offset;
  compress_offset_->compute(output.bucket_range, batch_size_per_gpu_, &num_keys_per_lookup_offset);

  indices_converter_->convert(output.keys, output.h_num_keys, num_keys_per_lookup_offset,
                              d_local_table_ids_);
}

SparseMPDataDistributionOp::MPTempStorage::MPTempStorage(
    std::shared_ptr<core::CoreResourceManager> core, int batch_size, int sample_max_nnz,
    int max_local_features, int max_local_buckets, int max_buckets_in_group,
    core23::DataType key_type, core23::DataType offset_type) {
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
      auto num_items = batch_size_per_dev * sample_max_nnz;
      cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                      (KeyType*)nullptr, (KeyType*)nullptr,
                                      static_cast<int64_t>(num_items));
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

  this->k_per_b_feat_major = core23::Tensor(
      params.shape({static_cast<int64_t>(batch_size * max_local_buckets)}).data_type(offset_type));
  DISPATCH_INTEGRAL_FUNCTION_CORE23(offset_type.type(), offset_t, [&] {
    HCTR_LIB_THROW(cudaMallocHost((void**)&this->h_send_k_per_g,
                                  core->get_global_gpu_count() * sizeof(offset_t)));
    HCTR_LIB_THROW(cudaMallocHost((void**)&this->h_recv_k_per_g,
                                  core->get_global_gpu_count() * sizeof(offset_t)));
  });
}

SparseMPDataDistributionOp::SparseMPDataDistributionOp(
    std::shared_ptr<core::CoreResourceManager> core,
    const embedding::EmbeddingCollectionParam& ebc_param, size_t group_id,
    const std::vector<embedding::EmbeddingTableParam>& emb_table_param_list)
    : core_(core),
      ebc_param_(ebc_param),
      num_global_gpus_(core->get_global_gpu_count()),
      batch_size_per_gpu_(ebc_param.universal_batch_size / num_global_gpus_),
      label_and_count_keys_operator_(core, ebc_param, group_id),
      label_and_count_keys_output_(core, ebc_param, group_id),
      count_keys_operator_(core, ebc_param, group_id),
      transpose_buckets_operator_(core, ebc_param, group_id),
      swizzle_keys_operator_(core, ebc_param, group_id) {
  // ---- allocate temp storage ----
  sample_max_nnz_ = 0;
  for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
    sample_max_nnz_ += ebc_param_.lookup_params[lookup_id].max_hotness;
  }

  CudaDeviceContext context(core_->get_device_id());

  int max_local_features = 0;
  int max_local_buckets = 0;
  int max_buckets_in_group = 0;
  for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
    if (ebc_param_.has_table_shard(core->get_global_gpu_id(), group_id, lookup_id)) {
      max_local_features += ebc_param_.lookup_params[lookup_id].max_hotness;
      max_local_buckets++;
    }
    if (ebc_param_.lookup_id_in_group(group_id, lookup_id)) {
      max_buckets_in_group++;
    }
  }
  sparse_temp_storage_ = MPTempStorage(core, ebc_param_.universal_batch_size, sample_max_nnz_,
                                       max_local_features, max_local_buckets, max_buckets_in_group,
                                       ebc_param_.key_type, ebc_param_.offset_type);
  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::AddOffset) {
    indices_converter_ = std::make_unique<embedding::KeysToIndicesConverter>(
        core, emb_table_param_list, ebc_param_, group_id);

    std::vector<int> h_local_lookup_id_list;
    std::vector<int> h_local_table_id_list;

    for (int lookup_id = 0; lookup_id < ebc_param_.num_lookup; ++lookup_id) {
      if (!ebc_param_.has_table_shard(core->get_global_gpu_id(), group_id, lookup_id)) continue;
      int table_id = ebc_param_.lookup_params[lookup_id].table_id;

      h_local_lookup_id_list.push_back(lookup_id);
      h_local_table_id_list.push_back(table_id);
    }
    compress_offset_ = std::make_unique<embedding::CompressOffset>(
        core, h_local_lookup_id_list.size() + 1, ebc_param_.offset_type);

    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    d_local_table_ids_ =
        core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_id_list.size())})
                           .data_type(core23::ScalarType::Int32));
    core23::copy_sync(d_local_table_ids_, h_local_table_id_list);
  }
}

void SparseMPDataDistributionOp::distribute(const DataDistributionInput& input,
                                            embedding::EmbeddingInput& output, int batch_size,
                                            cudaStream_t stream) {
  filter_before_all2all(input, output, stream);
  all2all_keys_per_bucket(output, stream);
  all2all_keys(output, stream);
  filter_after_all2all(output, stream);
  convert_indices(output);
}

void SparseMPDataDistributionOp::filter_before_all2all(const DataDistributionInput& input,
                                                       embedding::EmbeddingInput& output,
                                                       cudaStream_t stream) {
  // --- Label keys for sort, and count keys per bucket & GPU ---
  label_and_count_keys_operator_(input, label_and_count_keys_output_, stream);
}

void SparseMPDataDistributionOp::all2all_keys_per_bucket(embedding::EmbeddingInput& output,
                                                         cudaStream_t stream) {
  const auto& per_gpu_lookup_range = label_and_count_keys_operator_.h_per_gpu_lookup_range;
  auto send_tensor = label_and_count_keys_output_.keys_per_bucket;
  auto recv_tensor = sparse_temp_storage_.k_per_b_gpu_major;
  auto nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensor.data_type().type());

  size_t recv_num_buckets =
      per_gpu_lookup_range[(core_->get_global_gpu_id() + 1) * ebc_param_.num_lookup] -
      per_gpu_lookup_range[core_->get_global_gpu_id() * ebc_param_.num_lookup];

  size_t recv_offset = 0;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_tensor.data_type().type(), BucketRangeType, [&] {
    ncclGroupStart();
    for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
      size_t start_range = per_gpu_lookup_range[peer * ebc_param_.num_lookup];
      size_t send_num_buckets =
          per_gpu_lookup_range[(peer + 1) * ebc_param_.num_lookup] - start_range;

      HCTR_LIB_THROW(ncclSend(send_tensor.data<BucketRangeType>() + start_range, send_num_buckets,
                              nccl_type, peer, core_->get_nccl(), stream));
      HCTR_LIB_THROW(ncclRecv(recv_tensor.data<BucketRangeType>() + recv_offset, recv_num_buckets,
                              nccl_type, peer, core_->get_nccl(), stream));

      recv_offset += recv_num_buckets;
    }
    ncclGroupEnd();
  });
}

void SparseMPDataDistributionOp::all2all_keys(embedding::EmbeddingInput& output,
                                              cudaStream_t stream) {
  // --- count keys per gpu, so we know how many keys to receive from nccl
  // TODO: compare with cub::DeviceSegmentedReduce::Sum
  {
    auto k_per_b_gpu_major = sparse_temp_storage_.k_per_b_gpu_major;
    auto k_per_g = sparse_temp_storage_.k_per_g;

    count_keys_operator_(k_per_b_gpu_major, k_per_g, stream);
  }

  // --- sort keys by global gpu id ---
  DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.key_type.type(), KeyType, [&] {
    size_t temp_bytes = sparse_temp_storage_.temp_sort_storage.num_bytes();
    void* temp_ptr = sparse_temp_storage_.temp_sort_storage.data();
    auto keys_in = label_and_count_keys_output_.flat_keys.data<KeyType>();
    auto keys_out = sparse_temp_storage_.sorted_local_keys.data<KeyType>();
    auto labels_in = label_and_count_keys_output_.local_labels.data<uint32_t>();
    auto labels_out = sparse_temp_storage_.sorted_local_labels.data<uint32_t>();

    // ATTENTION: cub radix sort requires NumItemT to be consistent
    int sort_end_bit = (int)log2(num_global_gpus_) + 1;

    cub::DeviceRadixSort::SortPairs(temp_ptr, temp_bytes, labels_in, labels_out, keys_in, keys_out,
                                    static_cast<int64_t>(batch_size_per_gpu_ * sample_max_nnz_), 0,
                                    sort_end_bit, stream);
  });

  // --- Inter-node traffic ---
  auto send_tensor = sparse_temp_storage_.sorted_local_keys;
  auto recv_tensor = sparse_temp_storage_.keys;

  auto send_k_per_g = label_and_count_keys_output_.keys_per_gpu;
  auto recv_k_per_g = sparse_temp_storage_.k_per_g;

  auto nccl_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(send_tensor.data_type().type());

  // Prefetch counts to host
  void* h_send_k_per_g = sparse_temp_storage_.h_send_k_per_g;
  void* h_recv_k_per_g = sparse_temp_storage_.h_recv_k_per_g;
  HCTR_LIB_THROW(cudaMemcpyAsync(h_send_k_per_g, send_k_per_g.data(), send_k_per_g.num_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaMemcpyAsync(h_recv_k_per_g, recv_k_per_g.data(), recv_k_per_g.num_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  size_t send_offset = 0;
  size_t recv_offset = 0;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(send_tensor.data_type().type(), KeyType, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(send_k_per_g.data_type().type(), BucketRangeType, [&] {
      ncclGroupStart();
      for (size_t peer = 0; peer < num_global_gpus_; ++peer) {
        auto send_num_keys = static_cast<BucketRangeType*>(h_send_k_per_g)[peer];
        auto recv_num_keys = static_cast<BucketRangeType*>(h_recv_k_per_g)[peer];
        //        printf("GPU (%d) -> (%d) send_num_keys: %d, recv_num_keys: %d\n",
        //        core->get_global_gpu_id(),
        //               (int)peer, (int)send_num_keys, (int)recv_num_keys);
        if (send_num_keys > 0) {
          HCTR_LIB_THROW(ncclSend(send_tensor.data<KeyType>() + send_offset, send_num_keys,
                                  nccl_type, peer, core_->get_nccl(), stream));
        }
        if (recv_num_keys > 0) {
          HCTR_LIB_THROW(ncclRecv(recv_tensor.data<KeyType>() + recv_offset, recv_num_keys,
                                  nccl_type, peer, core_->get_nccl(), stream));
        }
        send_offset += send_num_keys;
        recv_offset += recv_num_keys;
      }
      ncclGroupEnd();
    });
  });

  output.h_num_keys = recv_offset;
}

void SparseMPDataDistributionOp::filter_after_all2all(embedding::EmbeddingInput& output,
                                                      cudaStream_t stream) {
  // --- computes bucket ranges received from nccl
  DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
    size_t temp_bytes = sparse_temp_storage_.temp_scan_storage.num_bytes();
    void* temp_ptr = sparse_temp_storage_.temp_scan_storage.data();
    auto k_per_b_gpu_major = sparse_temp_storage_.k_per_b_gpu_major;
    auto bucket_range_gpu_major = sparse_temp_storage_.bucket_range_gpu_major;

    // computes in-place!
    HCTR_LIB_THROW(cudaMemsetAsync(bucket_range_gpu_major.data<BucketRangeType>(), 0,
                                   sizeof(BucketRangeType), stream));
    cub::DeviceScan::InclusiveSum(temp_ptr, temp_bytes, k_per_b_gpu_major.data<BucketRangeType>(),
                                  bucket_range_gpu_major.data<BucketRangeType>() + 1,
                                  k_per_b_gpu_major.num_elements(), stream);
  });

  {
    transpose_buckets_operator_(sparse_temp_storage_.k_per_b_gpu_major,
                                sparse_temp_storage_.k_per_b_feat_major, stream);
  }

  // --- computes output bucket range
  DISPATCH_INTEGRAL_FUNCTION_CORE23(ebc_param_.offset_type.type(), BucketRangeType, [&] {
    size_t temp_bytes = sparse_temp_storage_.temp_scan_storage.num_bytes();
    void* temp_ptr = sparse_temp_storage_.temp_scan_storage.data();
    auto k_per_b_feat_major = sparse_temp_storage_.k_per_b_feat_major;

    HCTR_LIB_THROW(cudaMemsetAsync(output.bucket_range.data<BucketRangeType>(), 0,
                                   sizeof(BucketRangeType), stream));
    cub::DeviceScan::InclusiveSum(temp_ptr, temp_bytes, k_per_b_feat_major.data<BucketRangeType>(),
                                  output.bucket_range.data<BucketRangeType>() + 1,
                                  k_per_b_feat_major.num_elements(), stream);
  });

  // --- swizzle keys from gpu-major to feature-major
  {
    auto keys_in = sparse_temp_storage_.keys;
    auto bucket_range_in = sparse_temp_storage_.bucket_range_gpu_major;
    auto bucket_range_out = output.bucket_range;

    swizzle_keys_operator_(bucket_range_in, bucket_range_out, keys_in, output.keys, stream);
  }
}

void SparseMPDataDistributionOp::convert_indices(embedding::EmbeddingInput& output) {
  if (ebc_param_.keys_preprocess_strategy_ == embedding::KeysPreprocessStrategy::None) return;
  int batch_size = batch_size_per_gpu_ * num_global_gpus_;

  core23::Tensor num_keys_per_lookup_offset;
  compress_offset_->compute(output.bucket_range, batch_size, &num_keys_per_lookup_offset);

  indices_converter_->convert(output.keys, output.h_num_keys, num_keys_per_lookup_offset,
                              d_local_table_ids_);
}
}  // namespace HugeCTR
