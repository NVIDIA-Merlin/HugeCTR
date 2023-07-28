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
#include <embedding/data_distributor/data_distributor.hpp>
#include <embedding/operators/communication.hpp>

namespace HugeCTR {

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
    h_hotness.push_back(lookup_params[lookup_id].max_hotness);

    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;
    h_local_lookup_ids.push_back(lookup_id);
    h_local_hotness.push_back(lookup_params[lookup_id].max_hotness);

    if (ebc_param.grouped_lookup_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      int table_id = lookup_params[lookup_id].table_id;
      int shard_id, num_shard;
      ebc_param.get_table_shard_id(gpu_id, table_id, &shard_id, &num_shard);

      h_local_shard_ids.push_back(shard_id);
      h_local_num_shards.push_back(num_shard);
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

    if (ebc_param.grouped_lookup_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
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
  size_t num_local_gpus = core_resource_managers_.size();
  for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    std::vector<KeyFilterInitParams> init_params_for_current_gpu;
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size();
         ++grouped_id) {
      init_params_for_current_gpu.emplace_back(core_resource_managers_[local_gpu_id], ebc_param_,
                                               grouped_id);
    }
    key_filters_init_params_.push_back(init_params_for_current_gpu);
  }
  for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    CudaDeviceContext context(core_resource_managers_[local_gpu_id]->get_device_id());

    std::vector<KeyFilter> key_filters_for_current_gpu;
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size();
         ++grouped_id) {
      auto& grouped_emb_param = ebc_param_.grouped_lookup_params[grouped_id];

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
  if (ebc_param_.input_layout_ != embedding::EmbeddingLayout::BatchMajor) return;
  preprocess_inputs_.clear();

  size_t num_local_gpus = core_resource_managers_.size();
  for (size_t local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    CudaDeviceContext context(core_resource_managers_[local_gpu_id]->get_device_id());

    preprocess_inputs_.push_back(std::make_unique<embedding::PreprocessInput>(
        core_resource_managers_[local_gpu_id], ebc_param_));
  }
}

void DataDistributor::init_indices_converter() {
  if (ebc_param_.keys_preprocess_strategy_ != embedding::KeysPreprocessStrategy::AddOffset) return;
  int num_gpus = core_resource_managers_.size();
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    CudaDeviceContext context(core_resource_managers_[gpu_id]->get_device_id());
    int ggpu_id = core_resource_managers_[gpu_id]->get_global_gpu_id();
    for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size();
         grouped_id++) {
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

void DataDistributor::convert_indices(int gpu_id, DataDistributor::Result& output) {
  if (indices_converters_.empty()) return;
  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size(); ++grouped_id) {
    if (ebc_param_.grouped_lookup_params[grouped_id].grouped_table_idx == -1) continue;

    int batch_size_per_gpu = batch_size_;
    if (ebc_param_.grouped_lookup_params[grouped_id].table_placement_strategy ==
        embedding::TablePlacementStrategy::DataParallel) {
      batch_size_per_gpu /= num_global_gpus_;
    }

    int num_groups = ebc_param_.grouped_lookup_params.size();
    core23::Tensor num_keys_per_lookup_offset;
    compress_offsets_[gpu_id * num_groups + grouped_id].compute(
        output[grouped_id].bucket_range, batch_size_per_gpu, &num_keys_per_lookup_offset);

    indices_converters_[gpu_id * num_groups + grouped_id].convert(
        output[grouped_id].keys, output[grouped_id].h_num_keys, num_keys_per_lookup_offset,
        d_local_table_id_lists_[gpu_id * num_groups + grouped_id]);
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

  HCTR_CHECK(ebc_param_.input_layout_ == embedding::EmbeddingLayout::BatchMajor);
  auto& preprocess_input = preprocess_inputs_[gpu_id];
  core23::Tensor feature_major_key, feature_major_bucket_range;
  preprocess_input->compute(fullbatch_keys, fullbatch_bucket_range, &feature_major_key,
                            &feature_major_bucket_range, batch_size);

  for (size_t grouped_id = 0; grouped_id < ebc_param_.grouped_lookup_params.size(); ++grouped_id) {
    auto& grouped_emb_param = ebc_param_.grouped_lookup_params[grouped_id];

    HCTR_THROW_IF(grouped_emb_param.embedding_type != embedding::EmbeddingType::Sparse,
                  Error_t::IllegalCall,
                  "data distributor does not support non-sparse embedding type");

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

    if (grouped_emb_param.table_placement_strategy ==
        embedding::TablePlacementStrategy::ModelParallel) {
      key_filters_[gpu_id][grouped_id].mp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else if (grouped_emb_param.table_placement_strategy ==
               embedding::TablePlacementStrategy::DataParallel) {
      key_filters_[gpu_id][grouped_id].dp_index_calculation.filter_sparse_input(
          feature_major_key, feature_major_bucket_range, output[grouped_id], batch_size);
    } else {
      HCTR_OWN_THROW(Error_t::IllegalCall, "unsupported table placement strategy.");
    }
  }

  convert_indices(gpu_id, output);
}
}  // namespace HugeCTR
