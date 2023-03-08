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
#include <HugeCTR/embedding/view.hpp>
#include <HugeCTR/include/utils.cuh>

#include "keys_to_indices.hpp"
using namespace core;

namespace embedding {
template <typename key_t>
__global__ void keys_to_indices_kernel(key_t *keys, size_t num_keys,
                                       const uint32_t *num_keys_per_lookup_offset, int num_lookups,
                                       const int *table_id_list, const int *local_table_ids,
                                       int num_local_table_ids,
                                       const uint64_t *num_keys_per_table_offset,
                                       const int *num_shards, int gpu_id) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, tid, num_keys) {
    int table_id_idx = bs_upper_bound_sub_one(num_keys_per_lookup_offset, num_lookups + 1, tid);

    int table_id = table_id_list[table_id_idx];
    int local_table_id_idx = bs_upper_bound_sub_one(local_table_ids, num_local_table_ids, table_id);

    uint64_t start = num_keys_per_table_offset[local_table_id_idx];
    key_t k = keys[tid];

    uint64_t idx = k / (num_shards == nullptr ? 1 : num_shards[table_id]);

    keys[tid] = static_cast<key_t>(start + idx);
  }
}

KeysToIndicesConverter::KeysToIndicesConverter(std::shared_ptr<CoreResourceManager> core,
                                               const std::vector<EmbeddingTableParam> &table_params,
                                               const EmbeddingCollectionParam &ebc_param,
                                               size_t grouped_id)
    : core_(core) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int global_gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  const auto &emb_param = ebc_param.grouped_emb_params[grouped_id];

  std::vector<uint64_t> h_num_keys_per_table_offset{0};
  if (emb_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
    for (int table_id : emb_param.table_ids) {
      h_local_table_ids_.push_back(table_id);
      h_num_keys_per_table_offset.push_back(table_params[table_id].max_vocabulary_size);
    }
  } else if (emb_param.table_placement_strategy == TablePlacementStrategy::ModelParallel) {
    h_num_shards_.resize(ebc_param.shard_matrix[0].size());
    for (int table_id : emb_param.table_ids) {
      if (ebc_param.shard_matrix[global_gpu_id][table_id] == 0) continue;
      h_local_table_ids_.push_back(table_id);

      std::vector<int> shard_gpu_list;
      for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        HCTR_CHECK_HINT(table_id < static_cast<int>(ebc_param.shard_matrix[gpu_id].size()),
                        "table_id is out of range");
        if (ebc_param.shard_matrix[gpu_id][table_id] == 1) {
          shard_gpu_list.push_back(gpu_id);
        }
      }

      int num_shards = static_cast<int>(shard_gpu_list.size());
      h_num_shards_[table_id] = num_shards;
      auto find_shard_id_iter =
          std::find(shard_gpu_list.begin(), shard_gpu_list.end(), global_gpu_id);
      HCTR_CHECK(find_shard_id_iter != shard_gpu_list.end());
      int shard_id = static_cast<int>(std::distance(shard_gpu_list.begin(), find_shard_id_iter));

      uint64_t num_keys =
          table_params[table_id].max_vocabulary_size / num_shards +
          (shard_id < table_params[table_id].max_vocabulary_size % num_shards ? 1 : 0);
      h_num_keys_per_table_offset.push_back(num_keys);
    }
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                   "Unspecified table placement strategy in KeysToIndicesConverter.");
  }

  std::partial_sum(h_num_keys_per_table_offset.begin(), h_num_keys_per_table_offset.end(),
                   h_num_keys_per_table_offset.begin());

  core23::Device device(core23::DeviceType::GPU, core_->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  local_table_ids_ = core23::Tensor(params.shape({static_cast<int64_t>(h_local_table_ids_.size())})
                                        .data_type(core23::ScalarType::Int32));
  num_keys_per_table_offset_ =
      core23::Tensor(params.shape({static_cast<int64_t>(h_num_keys_per_table_offset.size())})
                         .data_type(core23::ScalarType::UInt64));

  core23::copy_sync(local_table_ids_, h_local_table_ids_);
  core23::copy_sync(num_keys_per_table_offset_, h_num_keys_per_table_offset);
  if (!h_num_shards_.empty()) {
    core23::TensorParams num_shards_params = core23::TensorParams().device(device);
    num_shards_ =
        core23::Tensor(num_shards_params.shape({static_cast<int64_t>(h_num_shards_.size())})
                           .data_type(core23::ScalarType::Int32));
    core23::copy_sync(num_shards_, h_num_shards_);
  }
}

void KeysToIndicesConverter::convert(core23::Tensor &keys, size_t num_keys,
                                     const core23::Tensor &num_keys_per_lookup_offset,
                                     size_t num_lookups, const core23::Tensor &table_id_list) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  if (num_keys > 0) {  // batch size is small there can be situation that we do not need have
    DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), key_t, [&] {
      // key for lookup
      constexpr int block_size = 256;
      int grid_size = (num_keys - 1) / block_size + 1;
      keys_to_indices_kernel<<<grid_size, block_size, 0, stream>>>(
          keys.data<key_t>(), num_keys, num_keys_per_lookup_offset.data<uint32_t>(), num_lookups,
          table_id_list.data<int>(), local_table_ids_.data<int>(), local_table_ids_.num_elements(),
          num_keys_per_table_offset_.data<uint64_t>(),
          !h_num_shards_.empty() ? num_shards_.data<int>() : nullptr, core_->get_global_gpu_id());
    });
  }
}
}  // namespace embedding