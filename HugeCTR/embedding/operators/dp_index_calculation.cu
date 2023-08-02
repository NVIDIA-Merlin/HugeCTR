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

#include <HugeCTR/embedding/operators/communication.hpp>
#include <cub/cub.cuh>
#include <embedding/data_distributor/data_compression_operators.cuh>
#include <embedding/operators/dp_index_calculation.hpp>
#include <utils.cuh>

namespace embedding {
namespace {

template <typename key_t>
__global__ void cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel(
    const key_t* unique_indices, int num_elements,
    const uint32_t* ev_start_indices_in_allreduce_buffer, const size_t* num_unique_key,
    uint32_t* ev_start_indices_for_local_reduce) {
  uint32_t num_keys = static_cast<uint32_t>(*num_unique_key);
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_elements) {
    if (i >= num_keys) {
      ev_start_indices_for_local_reduce[i] = 0;
      continue;
    }
    uint32_t idx = i;

    int idx_in_allreduce_buffer = static_cast<int>(unique_indices[idx]);

    ev_start_indices_for_local_reduce[i] =
        ev_start_indices_in_allreduce_buffer[idx_in_allreduce_buffer];
  }
}
}  // namespace

void DenseAllreduceIndexCalculation::cal_for_sparse_indices(
    const EmbeddingInput& embedding_input,
    const core23::Tensor& ev_start_indices_in_allreduce_buffer, ReductionIndices& reduction_indices,
    Wgrad& wgrad, int batch_size_per_gpu) {
  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, indices_sort_,
                                                       segmented_unique_, reduction_indices, wgrad,
                                                       batch_size_per_gpu);
  if (embedding_input.h_num_keys == 0) return;
  auto key_type = wgrad.unique_keys.data_type();
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel<<<144 * 8, 256, 0, stream>>>(
        wgrad.unique_keys.data<key_t>(), wgrad.unique_keys.num_elements(),
        ev_start_indices_in_allreduce_buffer.data<uint32_t>(), wgrad.num_unique_keys.data<size_t>(),
        wgrad.ev_start_indices.data<uint32_t>());
  });
}

namespace {

template <typename KeyType>
using TableEntry = HugeCTR::TableEntry<KeyType>;

template <typename KeyType>
struct UniqueAndStoreLowestIdxTableView {
  using Hash = HugeCTR::Hash;
  using KeyPair = HugeCTR::KeyPair<KeyType>;
  using TableValue = HugeCTR::TableValue;

  TableEntry<KeyType>* table;
  size_t capacity;

  DEVICE_INLINE void insert(const KeyPair& key_pair, const uint32_t& idx) noexcept {
    const KeyType& key = key_pair.key;
    const KeyType key_hi = (key | 0x1);
    const uint32_t key_lo = static_cast<uint32_t>(key & 0x1);
    const int& feature_id = key_pair.feature_id;
    size_t pos = Hash()(key_pair) % capacity;

    uint32_t idx_plus_one = idx + 1;
    uint32_t r_idx_plus_one = 0;
    while (r_idx_plus_one == 0) {
      bool prob_next = false;

      KeyType* key_ptr = &table[pos].key;
      volatile uint64_t* table_value_ptr = &table[pos].value.value;

      const KeyType old_key = atomicCAS(key_ptr, 0, key_hi);
      if (old_key == 0) {
        TableValue insert_value;
        insert_value.detail.r_idx_plus_one = idx_plus_one;
        insert_value.detail.feature_id_and_key_lo = (feature_id << 1U | key_lo);
        *table_value_ptr = insert_value.value;
        r_idx_plus_one = idx_plus_one;
      } else if (old_key == key_hi) {
        TableValue table_value;
        table_value.value = *table_value_ptr;
        uint32_t table_r_idx_plus_one = table_value.detail.r_idx_plus_one;
        uint32_t table_feature_id_and_key_lo = table_value.detail.feature_id_and_key_lo;

        if (table_r_idx_plus_one == 0 && table_feature_id_and_key_lo == 0) {
          // do nothing.
        } else if ((table_feature_id_and_key_lo & 0x1) == key_lo &&
                   (table_feature_id_and_key_lo >> 1U) == feature_id) {
          if (table_r_idx_plus_one > idx_plus_one) {
            // do substitution if the idx is smaller
            TableValue insert_value;
            insert_value.detail.r_idx_plus_one = idx_plus_one;
            insert_value.detail.feature_id_and_key_lo = (feature_id << 1U | key_lo);

            uint64_t old_table_value =
                atomicCAS((uint64_t*)table_value_ptr, table_value.value, insert_value.value);
            // table value has been changed, retry
            if (old_table_value == table_value.value) r_idx_plus_one = idx_plus_one;
          } else {
            // else return smaller idx
            r_idx_plus_one = table_r_idx_plus_one;
          }
        } else {
          prob_next = true;
        }
      } else {
        prob_next = true;
      }

      if (prob_next) {
        pos += 1;
        if (pos >= capacity) {
          pos -= capacity;
        }
      }
    }
  }

  DEVICE_INLINE uint32_t lookup(const KeyPair& key_pair) const noexcept {
    const KeyType& key = key_pair.key;
    const KeyType key_hi = (key | 0x1);
    const uint32_t key_lo = static_cast<uint32_t>(key & 0x1);
    const int& feature_id = key_pair.feature_id;
    size_t pos = Hash()(key_pair) % capacity;

    uint32_t r_idx = HugeCTR::kInvalidReverseIdx;
    while (r_idx == HugeCTR::kInvalidReverseIdx) {
      const KeyType old_key = table[pos].key;

      if (old_key == key_hi) {
        TableValue table_value;
        table_value.value = table[pos].value.value;
        uint32_t table_feature_id_and_key_lo = table_value.detail.feature_id_and_key_lo;

        if ((table_feature_id_and_key_lo & 0x1) == key_lo &&
            (table_feature_id_and_key_lo >> 1U) == feature_id) {
          r_idx = table_value.detail.r_idx_plus_one;
        }
      }
      pos += 1;
      if (pos >= capacity) {
        pos -= capacity;
      }
    }
    return r_idx;
  }
};

template <typename KeyType>
__global__ void insert_allgather_keys_into_hash_table_kernel(
    const KeyType* keys, const int* table_ids, size_t num_keys,
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    const KeyType key = keys[i];
    const int table_id = table_ids[i];

    hash_table.insert({key, table_id}, i);
  }
}

template <typename KeyType>
__global__ void insert_unique_keys_and_ev_start_indices_into_hash_table_kernel(
    const KeyType* keys, const int* table_ids, const uint32_t* ev_start_indices, uint64_t* num_keys,
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table) {
  CUDA_1D_KERNEL_LOOP(i, *num_keys) {
    const KeyType key = keys[i];
    const int table_id = table_ids[i];
    const uint32_t ev_start_indice = ev_start_indices[i];

    hash_table.insert({key, table_id}, ev_start_indice);
  }
}

template <typename KeyType>
__global__ void mask_unique_keys_in_allgather_keys_kernel(
    const KeyType* keys, const int* table_ids, size_t num_keys,
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table, int* mask) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    const KeyType key = keys[i];
    const int table_id = table_ids[i];

    uint32_t idx_plus_one = hash_table.lookup({key, table_id});
    mask[i] = (idx_plus_one == i + 1) ? 1 : 0;
  }
}

template <typename KeyType>
__global__ void lookup_ev_start_indices_in_hash_table_kernel(
    const KeyType* keys, const int* table_ids, uint64_t* num_keys,
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table, uint32_t* ev_start_indices) {
  CUDA_1D_KERNEL_LOOP(i, *num_keys) {
    const KeyType key = keys[i];
    const int table_id = table_ids[i];

    uint32_t ev_start_indices_plus_one = hash_table.lookup({key, table_id});
    ev_start_indices[i] = ev_start_indices_plus_one - 1;
  }
}

__global__ void table_id_to_ev_size_kernel(const int* table_ids, const uint64_t* num_keys,
                                           const int* table_id_to_ev_size,
                                           uint32_t* ev_sizes_of_unqiue_keys) {
  CUDA_1D_KERNEL_LOOP(i, *num_keys) {
    ev_sizes_of_unqiue_keys[i] = table_id_to_ev_size[table_ids[i]];
  }
}
}  // namespace

SparseAllreduceCalEVStartIndicesStorage::SparseAllreduceCalEVStartIndicesStorage(
    std::shared_ptr<CoreResourceManager> core, int local_hotness_sum, int batch_size_per_gpu,
    core23::DataType key_type) {
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  int num_gpus = core->get_global_gpu_count();

  // BroadcastResult
  broadcast_result_.allgather_num_unique_keys_ =
      core23::Tensor(params.shape({num_gpus}).data_type(core23::ScalarType::UInt64));
  broadcast_result_.h_allgather_num_unique_keys_ =
      core23::Tensor(params.shape({num_gpus})
                         .data_type(core23::ScalarType::UInt64)
                         .device(core23::DeviceType::CPU));

  broadcast_result_.allgather_unique_keys_ = core23::Tensor(
      params.shape({local_hotness_sum * batch_size_per_gpu * num_gpus}).data_type(key_type));
  broadcast_result_.allgather_table_ids_ =
      core23::Tensor(params.shape({local_hotness_sum * batch_size_per_gpu * num_gpus})
                         .data_type(core23::ScalarType::Int32));
  broadcast_result_.num_allgather_keys_ = 0ul;

  // HashTable
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    hash_table_ =
        core23::Tensor(params
                           .shape({static_cast<int64_t>(local_hotness_sum * batch_size_per_gpu *
                                                        num_gpus * sizeof(TableEntry<key_t>))})
                           .data_type(core23::ScalarType::Char));
  });

  // Tempstorage
  temp_storage_.mask_unique_keys_in_allgather_unique_keys_ =
      core23::Tensor(params.shape({local_hotness_sum * batch_size_per_gpu * num_gpus})
                         .data_type(core23::ScalarType::Int32));
  {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_bytes, (int64_t*)nullptr, (int*)nullptr,
                               (int64_t*)nullptr, (int64_t*)nullptr,
                               local_hotness_sum * batch_size_per_gpu * num_gpus);
    temp_storage_.d_temp_select_unique_keys_in_allgather_unique_keys_ = core23::Tensor(
        params.shape({static_cast<int64_t>(temp_bytes)}).data_type(core23::ScalarType::Char));
  }
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  local_hotness_sum * batch_size_per_gpu * num_gpus + 1);
    temp_storage_.d_temp_scan_ev_start_indices_storage_ = core23::Tensor(
        params.shape({static_cast<int64_t>(temp_bytes)}).data_type(core23::ScalarType::Char));
  }
}

void broadcast_unique_keys(const std::shared_ptr<core::CoreResourceManager>& core,
                           const Wgrad& local_reduce_wgrad, BroadcastResult& broadcast_result) {
  cudaStream_t stream = core->get_local_gpu()->get_stream();
  ncclComm_t comm = core->get_nccl();
  int num_gpus = core->get_global_gpu_count();

  // 1. collect num_unique keys
  HCTR_LIB_THROW(ncclAllGather(local_reduce_wgrad.num_unique_keys.data(),
                               broadcast_result.allgather_num_unique_keys_.data(), 1, ncclUint64,
                               comm, stream));
  core23::copy_async(broadcast_result.h_allgather_num_unique_keys_,
                     broadcast_result.allgather_num_unique_keys_, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  auto key_type = local_reduce_wgrad.unique_keys.data_type();
  auto nccl_key_type = core23::get_nccl_dtype_from_tensor_scalar_type_core23(key_type.type());

  // 2. broadcast unique keys and table ids
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    HCTR_LIB_THROW(ncclGroupStart());
    const uint64_t* h_allgather_num_unique_keys =
        broadcast_result.h_allgather_num_unique_keys_.data<uint64_t>();
    uint64_t count_offset = 0;

    for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
      uint64_t num_unique_keys = h_allgather_num_unique_keys[dst_gpu_id];

      HCTR_LIB_THROW(
          ncclBroadcast(local_reduce_wgrad.unique_keys.data<key_t>(),
                        broadcast_result.allgather_unique_keys_.data<key_t>() + count_offset,
                        num_unique_keys, nccl_key_type, dst_gpu_id, comm, stream));
      HCTR_LIB_THROW(ncclBroadcast(local_reduce_wgrad.table_ids.data<int>(),
                                   broadcast_result.allgather_table_ids_.data<int>() + count_offset,
                                   num_unique_keys, ncclInt32, dst_gpu_id, comm, stream));
      count_offset += num_unique_keys;
    }

    broadcast_result.num_allgather_keys_ = count_offset;
    HCTR_LIB_THROW(ncclGroupEnd());
  });
}

void cal_allreduce_wgrad(const std::shared_ptr<core::CoreResourceManager>& core,
                         const BroadcastResult& broadcast_result,
                         SparseAllreduceCalEVStartIndicesTempStorage& temp_storage,
                         core23::Tensor& unique_and_sort_hash_table, Wgrad& allreduce_wgrad) {
  cudaStream_t stream = core->get_local_gpu()->get_stream();
  auto key_type = broadcast_result.allgather_unique_keys_.data_type();
  HCTR_LIB_THROW(cudaMemsetAsync(unique_and_sort_hash_table.data(), 0,
                                 unique_and_sort_hash_table.num_bytes(), stream));
  size_t num_allgather_keys = broadcast_result.num_allgather_keys_;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    size_t capacity = unique_and_sort_hash_table.num_bytes() / sizeof(TableEntry<KeyType>);

    UniqueAndStoreLowestIdxTableView<KeyType> hash_table{
        (TableEntry<KeyType>*)unique_and_sort_hash_table.data(), capacity};
    auto& kernel_param = core->get_kernel_param();
    int block_size = 256;
    int grid_size = std::min(kernel_param.num_sms * (kernel_param.max_thread_per_sm / block_size),
                             static_cast<int>(num_allgather_keys / block_size));

    //  1. unique_and_sort_using_hash_table
    insert_allgather_keys_into_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(
        broadcast_result.allgather_unique_keys_.data<KeyType>(),
        broadcast_result.allgather_table_ids_.data<int>(), num_allgather_keys, hash_table);

    // 2. mask selected unique keys
    HCTR_LIB_THROW(cudaMemsetAsync(
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.data(), 0,
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.num_bytes(), stream));
    mask_unique_keys_in_allgather_keys_kernel<<<grid_size, block_size, 0, stream>>>(
        broadcast_result.allgather_unique_keys_.data<KeyType>(),
        broadcast_result.allgather_table_ids_.data<int>(), num_allgather_keys, hash_table,
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.data<int>());

    // 3. select unique keys / num_unique_keys / table_ids
    size_t select_unique_keys_temp_nbytes =
        temp_storage.d_temp_select_unique_keys_in_allgather_unique_keys_.num_bytes();
    cub::DeviceSelect::Flagged(
        temp_storage.d_temp_select_unique_keys_in_allgather_unique_keys_.data(),
        select_unique_keys_temp_nbytes, broadcast_result.allgather_unique_keys_.data<KeyType>(),
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.data<int>(),
        allreduce_wgrad.unique_keys.data<KeyType>(),
        allreduce_wgrad.num_unique_keys.data<uint64_t>(), num_allgather_keys, stream);
    cub::DeviceSelect::Flagged(
        temp_storage.d_temp_select_unique_keys_in_allgather_unique_keys_.data(),
        select_unique_keys_temp_nbytes, broadcast_result.allgather_table_ids_.data<int>(),
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.data<int>(),
        allreduce_wgrad.table_ids.data<int>(), allreduce_wgrad.num_unique_keys.data<uint64_t>(),
        num_allgather_keys, stream);

    // 4. ev_start_indices
    HCTR_LIB_THROW(cudaMemsetAsync(allreduce_wgrad.ev_start_indices.data(), 0,
                                   allreduce_wgrad.ev_start_indices.num_bytes(), stream));
    table_id_to_ev_size_kernel<<<grid_size, block_size, 0, stream>>>(
        allreduce_wgrad.table_ids.data<int>(), allreduce_wgrad.num_unique_keys.data<uint64_t>(),
        allreduce_wgrad.attr.table_id_to_ev_size.data<int>(),
        allreduce_wgrad.ev_start_indices.data<uint32_t>() + 1);

    // 5. cal_allreduce_ev_start_indices
    size_t scan_ev_start_indices_temp_nbytes =
        temp_storage.d_temp_scan_ev_start_indices_storage_.num_bytes();
    cub::DeviceScan::InclusiveSum(temp_storage.d_temp_scan_ev_start_indices_storage_.data(),
                                  scan_ev_start_indices_temp_nbytes,
                                  allreduce_wgrad.ev_start_indices.data<int>(),
                                  allreduce_wgrad.ev_start_indices.data<int>(),
                                  allreduce_wgrad.ev_start_indices.num_elements(), stream);
  });
}

void init_key_to_ev_start_indices_hash_table(const std::shared_ptr<core::CoreResourceManager>& core,
                                             const Wgrad& allreduce_wgrad,
                                             core23::Tensor& key_to_ev_start_indices_hash_table) {
  cudaStream_t stream = core->get_local_gpu()->get_stream();

  auto key_type = allreduce_wgrad.unique_keys.data_type();
  HCTR_LIB_THROW(cudaMemsetAsync(key_to_ev_start_indices_hash_table.data(), 0,
                                 key_to_ev_start_indices_hash_table.num_bytes(), stream));

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    size_t capacity = key_to_ev_start_indices_hash_table.num_bytes() / sizeof(TableEntry<KeyType>);
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table{
        (TableEntry<KeyType>*)key_to_ev_start_indices_hash_table.data(), capacity};
    auto& kernel_param = core->get_kernel_param();
    int block_size = 256;
    int grid_size = kernel_param.num_sms * (kernel_param.max_thread_per_sm / block_size);

    //  1. unique_and_sort_using_hash_table
    insert_unique_keys_and_ev_start_indices_into_hash_table_kernel<<<grid_size, block_size, 0,
                                                                     stream>>>(
        allreduce_wgrad.unique_keys.data<KeyType>(), allreduce_wgrad.table_ids.data<int>(),
        allreduce_wgrad.ev_start_indices.data<uint32_t>(),
        allreduce_wgrad.num_unique_keys.data<uint64_t>(), hash_table);
  });
}

void cal_local_reduce_ev_start_indices(const std::shared_ptr<core::CoreResourceManager>& core,
                                       const core23::Tensor& key_to_ev_start_indices_hash_table,
                                       Wgrad& local_reduce_wgrad) {
  cudaStream_t stream = core->get_local_gpu()->get_stream();
  auto key_type = local_reduce_wgrad.unique_keys.data_type();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), KeyType, [&] {
    size_t capacity = key_to_ev_start_indices_hash_table.num_bytes() / sizeof(TableEntry<KeyType>);
    UniqueAndStoreLowestIdxTableView<KeyType> hash_table{
        (TableEntry<KeyType>*)key_to_ev_start_indices_hash_table.data(), capacity};
    auto& kernel_param = core->get_kernel_param();
    int block_size = 256;
    int grid_size = kernel_param.num_sms * (kernel_param.max_thread_per_sm / block_size);

    //  1. unique_and_sort_using_hash_table
    lookup_ev_start_indices_in_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(
        local_reduce_wgrad.unique_keys.data<KeyType>(), local_reduce_wgrad.table_ids.data<int>(),
        local_reduce_wgrad.num_unique_keys.data<uint64_t>(), hash_table,
        local_reduce_wgrad.ev_start_indices.data<uint32_t>());
  });
}

void SparseAllreduceIndexCalculation::cal_for_sparse_input(const EmbeddingInput& embedding_input,
                                                           ReductionIndices& reduction_indices,
                                                           Wgrad& local_reduce_wgrad,
                                                           Wgrad& allreduce_wgrad,
                                                           int batch_size_per_gpu) {
  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, segmented_sort_device_,
                                                       segmented_unique_, reduction_indices,
                                                       local_reduce_wgrad, batch_size_per_gpu);
  if (embedding_input.h_num_keys == 0) return;
  broadcast_unique_keys(core_, local_reduce_wgrad, cal_ev_start_indices_storage_.broadcast_result_);

  cal_allreduce_wgrad(core_, cal_ev_start_indices_storage_.broadcast_result_,
                      cal_ev_start_indices_storage_.temp_storage_,
                      cal_ev_start_indices_storage_.hash_table_, allreduce_wgrad);

  init_key_to_ev_start_indices_hash_table(core_, allreduce_wgrad,
                                          cal_ev_start_indices_storage_.hash_table_);

  cal_local_reduce_ev_start_indices(core_, cal_ev_start_indices_storage_.hash_table_,
                                    local_reduce_wgrad);
}

}  // namespace embedding
