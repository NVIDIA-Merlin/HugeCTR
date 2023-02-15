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

#include <cub/cub.cuh>
#include <embedding/operators/dp_index_calculation.hpp>
#include <embedding/operators/generic_lookup.cuh>
#include <utils.cuh>
#include <utils.hpp>

namespace embedding {
namespace {

template <typename key_t>
__global__ void cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel(
    const key_t* unique_indices, int num_elements, const uint32_t* table_range,
    const int* unique_table_ids, const int* table_id_to_allreduce_buffer_start_indices,
    const uint32_t* ev_start_indices_in_allreduce_buffer, const size_t* num_unique_key,
    int vocabulary_size, uint32_t* ev_start_indices_for_local_reduce) {
  uint32_t num_keys = static_cast<uint32_t>(*num_unique_key);
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_elements) {
    if (i >= num_keys) {
      ev_start_indices_for_local_reduce[i] = 0;
      continue;
    }
    uint32_t idx = i;

    int table_id = unique_table_ids[idx];
    int idx_in_allreduce_buffer = static_cast<int>(unique_indices[idx]) +
                                  table_id_to_allreduce_buffer_start_indices[table_id];

    ev_start_indices_for_local_reduce[i] =
        ev_start_indices_in_allreduce_buffer[idx_in_allreduce_buffer];
  }
}
}  // namespace

void DenseAllreduceIndexCalculation::cal_for_sparse_indices(
    const EmbeddingInput& embedding_input, const Tensor& table_id_to_allreduce_buffer_start_indices,
    const Tensor& ev_start_indices_in_allreduce_buffer, ReductionIndices& reduction_indices,
    Wgrad& wgrad, int batch_size) {
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  auto cal_ev_start_indices_in_allreduce_wgrad =
      [&](const WgradEvStartIndicesCalculationInput& input,
          WgradEvStartIndicesCalculationOutput& output, cudaStream_t stream) {
        auto key_type = input.unique_keys.dtype();

        DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
          cal_ev_start_indices_in_allreduce_wgrad_using_indices_kernel<<<144 * 8, 256, 0, stream>>>(
              input.unique_keys.get<key_t>(), input.unique_keys.get_num_elements(),
              input.table_range.get<uint32_t>(), input.table_ids.get<int>(),
              table_id_to_allreduce_buffer_start_indices.get<int>(),
              ev_start_indices_in_allreduce_buffer.get<uint32_t>(),
              input.num_unique_keys.get<size_t>(),
              ev_start_indices_in_allreduce_buffer.get_num_elements() - 1,
              output.ev_start_indices.get<uint32_t>());
        });
      };

  local_reduce_index_calculation_.cal_for_sparse_input(embedding_input, indices_sort_,
                                                       segmented_unique_, cal_dst_ids_,
                                                       reduction_indices, wgrad, batch_size);
  local_reduce_index_calculation_.cal_dst_ev_start(wgrad, cal_ev_start_indices_in_allreduce_wgrad);
}

SparseAllreduceCalEVStartIndicesStorage::SparseAllreduceCalEVStartIndicesStorage(
    std::shared_ptr<CoreResourceManager> core, int num_table, int local_hotness_sum, int batch_size,
    DataType key_type) {
  auto buffer_ptr = core::GetBuffer(core);
  int num_gpus = core->get_global_gpu_count();

  // BroadcastResult
  broadcast_result_.allgather_table_range_ = buffer_ptr->reserve(
      {num_gpus * (num_table + 1)}, core::DeviceType::GPU, core::TensorScalarType::UInt32);
  broadcast_result_.h_table_range_ =
      buffer_ptr->reserve({num_table + 1}, core::DeviceType::CPU, core::TensorScalarType::UInt32);
  broadcast_result_.reordered_allgather_table_range_ = buffer_ptr->reserve(
      {num_gpus * num_table + 1}, core::DeviceType::GPU, core::TensorScalarType::UInt32);
  broadcast_result_.h_reordered_allgather_table_range_ = buffer_ptr->reserve(
      {num_gpus * num_table + 1}, core::DeviceType::CPU, core::TensorScalarType::UInt32);
  broadcast_result_.allgather_unique_keys_ = buffer_ptr->reserve(
      {local_hotness_sum * batch_size * num_gpus}, core::DeviceType::GPU, key_type);

  // HashTable
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    hash_table_.hash_table_ =
        buffer_ptr->reserve({local_hotness_sum * batch_size * num_gpus * sizeof(TableEntry<key_t>)},
                            core::DeviceType::GPU, core::TensorScalarType::Void);
  });
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  num_table + 1);
    hash_table_.d_temp_scan_table_range_storage_ =
        buffer_ptr->reserve({temp_bytes}, core::DeviceType::GPU, TensorScalarType::Void);
  }

  // Tempstorage
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  local_hotness_sum * batch_size * num_gpus + 1);
    temp_storage_.d_temp_scan_ev_start_indices_storage_ =
        buffer_ptr->reserve({temp_bytes}, core::DeviceType::GPU, TensorScalarType::Void);
  }
  temp_storage_.mask_unique_keys_in_allgather_unique_keys_ = buffer_ptr->reserve(
      {local_hotness_sum * batch_size * num_gpus}, core::DeviceType::GPU, TensorScalarType::Int32);
  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
      cub::DeviceSelect::Flagged(nullptr, temp_bytes, (key_t*)nullptr, (int*)nullptr,
                                 (key_t*)nullptr, (size_t*)nullptr,
                                 local_hotness_sum * batch_size * num_gpus + 1);
    });
    temp_storage_.d_temp_select_temp_storage_ =
        buffer_ptr->reserve({temp_bytes}, core::DeviceType::GPU, TensorScalarType::Void);
  }
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                  local_hotness_sum * batch_size * num_gpus + 1);
    temp_storage_.d_temp_scan_unique_idx_temp_storage_ =
        buffer_ptr->reserve({temp_bytes}, core::DeviceType::GPU, TensorScalarType::Void);
  }
  temp_storage_.unique_idx_ = buffer_ptr->reserve({local_hotness_sum * batch_size * num_gpus},
                                                  core::DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
}

namespace {
// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp {
  // Running prefix
  int running_total;
  // Constructor
  explicit __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ int operator()(int block_aggregate) {
    int old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <int TPB>
__global__ void reorder_allgather_table_range_kernel(const uint32_t* allgather_table_range,
                                                     int num_table, int num_gpus,
                                                     uint32_t* reordered_table_range) {
  typedef cub::BlockScan<uint32_t, TPB> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockPrefixCallbackOp prefix_op(0);

  CUDA_1D_KERNEL_LOOP(i, num_table * num_gpus + 1) {
    uint32_t num_unique_keys = 0;
    if (i > 0) {
      int gpu_id = (i - 1) % num_gpus;
      int table_id = (i - 1) / num_gpus;

      num_unique_keys = allgather_table_range[gpu_id * (num_table + 1) + table_id + 1] -
                        allgather_table_range[gpu_id * (num_table + 1) + table_id];
    }

    uint32_t start;
    BlockScan(temp_storage).InclusiveSum(num_unique_keys, start, prefix_op);
    __syncthreads();

    reordered_table_range[i] = start;
  }
}

struct DirectHash {
  DEVICE_INLINE uint32_t operator()(int32_t v) { return static_cast<uint32_t>(v); }
  DEVICE_INLINE uint32_t operator()(uint32_t v) { return static_cast<uint32_t>(v); }
  DEVICE_INLINE uint32_t operator()(int64_t v) { return static_cast<uint32_t>(v); }
  DEVICE_INLINE uint32_t operator()(uint64_t v) { return static_cast<uint32_t>(v); }
};

// this kernel does:
template <typename key_t, typename HASH>
__global__ void hash_table_insert_key_and_index_kernel(const key_t* allgather_unique_keys,
                                                       const uint32_t* allgather_table_range,
                                                       int num_table, int num_gpus,
                                                       TableEntry<key_t>* table,
                                                       uint32_t* unique_keys_table_range) {
  for (int ith_gpu = blockIdx.y; ith_gpu < num_gpus; ith_gpu += gridDim.y) {
    for (int ith_table = blockIdx.x; ith_table < num_table; ith_table += gridDim.x) {
      uint32_t range_table_start = allgather_table_range[ith_table * num_gpus + ith_gpu];
      uint32_t range_table_end = allgather_table_range[ith_table * num_gpus + ith_gpu + 1];
      uint32_t num_keys_in_table = range_table_end - range_table_start;
      uint32_t table_capacity = allgather_table_range[(ith_table + 1) * num_gpus] -
                                allgather_table_range[ith_table * num_gpus];

      auto current_table = table + allgather_table_range[ith_table * num_gpus];
      for (uint32_t i = threadIdx.x; i < num_keys_in_table; i += blockDim.x) {
        uint32_t idx = range_table_start + i;
        uint32_t idx_plus_one = idx + 1;
        key_t key = allgather_unique_keys[idx];
        uint32_t key_hash = HASH()(key);
        uint32_t pos = key_hash % table_capacity;

        const key_t key_hi = (key | 0x1);
        const auto key_lo = static_cast<uint32_t>(key & 0x1);
        bool finish_insert = false;
        while (!finish_insert) {
          bool prob_next = false;
          key_t* key_ptr = &current_table[pos].key;
          volatile uint32_t* table_value_ptr = &current_table[pos].value;

          const key_t old_key = atomicCAS(key_ptr, 0, key_hi);
          if (old_key == 0) {
            *table_value_ptr = (idx_plus_one << 1U | key_lo);
            atomicAdd(unique_keys_table_range + 1 + ith_table, 1);
            finish_insert = true;
          } else if (old_key == key_hi) {
            const uint32_t value = *table_value_ptr;
            if (value == 0) {
              // do nothing.
            } else if ((value & 0x1) == key_lo) {
              if ((value >> 1U) > idx_plus_one) {
                // substitution with smaller idx
                *table_value_ptr = (idx_plus_one << 1U | key_lo);
              } else {
                // old idx is smaller. do nothing
              }
              finish_insert = true;
            } else {
              prob_next = true;
            }
          } else {
            prob_next = true;
          }
          if (prob_next) {
            pos += 1;
            if (pos >= table_capacity) {
              pos -= table_capacity;
            }
          }
        }
      }
    }
  }
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    unique_keys_table_range[0] = 0;
  }
}

template <typename key_t>
__global__ void hash_table_dump_index_mask_kernel(const TableEntry<key_t>* table,
                                                  const uint32_t* allgather_table_range,
                                                  int num_gpus, int num_table, int* mask) {
  for (int ith_table = blockIdx.x; ith_table < num_table; ith_table += gridDim.x) {
    uint32_t table_capacity = allgather_table_range[(ith_table + 1) * num_gpus] -
                              allgather_table_range[ith_table * num_gpus];

    auto current_table = table + allgather_table_range[ith_table * num_gpus];
    for (uint32_t i = threadIdx.x; i < table_capacity; i += blockDim.x) {
      const key_t key_hi = current_table[i].key;
      if (key_hi == 0) continue;
      const uint32_t value = current_table[i].value;
      uint32_t idx = ((value >> 1U) - 1);
      mask[idx] = 1;
    }
  }
}

__global__ void table_range_to_table_ids_and_ev_start_indices_kernel(
    int num_table, const uint32_t* table_range, const int* unique_table_ids,
    const int* table_id_to_ev_size, int* table_ids, uint32_t* ev_start_indices) {
  for (int ith_table = blockIdx.x; ith_table < num_table; ith_table += gridDim.x) {
    int table_id = unique_table_ids[ith_table];
    int ev_size = table_id_to_ev_size[table_id];

    uint32_t start = table_range[ith_table];
    uint32_t end = table_range[ith_table + 1];
    for (uint32_t i = threadIdx.x; i < (end - start); i += blockDim.x) {
      table_ids[start + i] = table_id;
      ev_start_indices[1 + start + i] = ev_size;
    }
  }
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    ev_start_indices[0] = 0;
  }
}

template <typename key_t, typename HASH>
__global__ void hash_table_lookup_key_and_map_ev_start_indices(
    const key_t* local_reduce_unique_keys, const uint32_t* local_reduce_table_range, int num_table,
    int num_gpus, const TableEntry<key_t>* table, const uint32_t* allgather_table_range,
    const int* unique_idx, const uint32_t* table_range, const uint32_t* allreduce_ev_start_indices,
    uint32_t* ev_start_indices) {
  for (int ith_table = blockIdx.x; ith_table < num_table; ith_table += gridDim.x) {
    uint32_t num_keys =
        local_reduce_table_range[ith_table + 1] - local_reduce_table_range[ith_table];
    uint32_t table_capacity = allgather_table_range[(ith_table + 1) * num_gpus] -
                              allgather_table_range[ith_table * num_gpus];

    auto current_local_reduce_unique_keys =
        local_reduce_unique_keys + local_reduce_table_range[ith_table];
    auto current_table = table + allgather_table_range[ith_table * num_gpus];
    auto current_ev_start_indices = ev_start_indices + local_reduce_table_range[ith_table];
    for (uint32_t i = threadIdx.x; i < num_keys; i += blockDim.x) {
      const key_t key = current_local_reduce_unique_keys[i];
      uint32_t key_hash = HASH()(key);
      uint32_t pos = key_hash % table_capacity;
      bool prob_next = true;
      while (prob_next) {
        const key_t key_hi = current_table[pos].key;
        const uint32_t value = current_table[pos].value;
        if (key == static_cast<key_t>((key_hi & ~(0x1)) | (value & 0x1))) {
          prob_next = false;
        } else {
          pos += 1;
          if (pos >= table_capacity) {
            pos -= table_capacity;
          }
        }
      }
      const uint32_t value = current_table[pos].value;
      uint32_t idx = (value >> 1U) - 1;
      current_ev_start_indices[i] = allreduce_ev_start_indices[unique_idx[idx] - 1];
    }
  }
}

}  // namespace

void broadcast_unique_keys(const embedding::WgradEvStartIndicesCalculationInput& input,
                           BroadcastResult& broadcast_result, int num_table, int num_gpus,
                           cudaStream_t stream, ncclComm_t comm) {
  // 1. collect table range
  HCTR_LIB_THROW(ncclAllGather(input.table_range.get<uint32_t>(),
                               broadcast_result.allgather_table_range_.get<uint32_t>(),
                               input.table_range.get_num_elements(), ncclUint32, comm, stream));
  input.table_range.copy_to(broadcast_result.h_table_range_, stream);

  // 2. calculate num_unique_keys on each gpu
  reorder_allgather_table_range_kernel<128><<<1, 128, 0, stream>>>(
      broadcast_result.allgather_table_range_.get<uint32_t>(), num_table, num_gpus,
      broadcast_result.reordered_allgather_table_range_.get<uint32_t>());
  broadcast_result.reordered_allgather_table_range_.copy_to(
      broadcast_result.h_reordered_allgather_table_range_, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  auto key_type = input.unique_keys.dtype();
  // 3. broadcast unique keys
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    HCTR_LIB_THROW(ncclGroupStart());
    const uint32_t* h_table_range_ptr = broadcast_result.h_table_range_.get<uint32_t>();
    const uint32_t* h_reordered_allgather_table_range_ptr =
        broadcast_result.h_reordered_allgather_table_range_.get<uint32_t>();

    for (int table_id = 0; table_id < num_table; ++table_id) {
      for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
        uint32_t num_unique_keys =
            h_reordered_allgather_table_range_ptr[table_id * num_gpus + dst_gpu_id + 1] -
            h_reordered_allgather_table_range_ptr[table_id * num_gpus + dst_gpu_id];
        HCTR_LIB_THROW(ncclBroadcast(
            input.unique_keys.get<key_t>() + h_table_range_ptr[table_id],
            broadcast_result.allgather_unique_keys_.get<key_t>() +
                h_reordered_allgather_table_range_ptr[table_id * num_gpus + dst_gpu_id],
            num_unique_keys, core::get_nccl_dtype_from_tensor_scalar_type(key_type.type()),
            dst_gpu_id, comm, stream));
      }
    }
    HCTR_LIB_THROW(ncclGroupEnd());
  });
}

void unique_broadcast_result_using_hash_table(
    const embedding::WgradEvStartIndicesCalculationInput& input,
    const BroadcastResult& broadcast_result, HashTable& hash_table, Wgrad& allreduce_wgrad,
    int num_table, int num_gpus, cudaStream_t stream) {
  auto key_type = input.unique_keys.dtype();
  // 4. cal table capacity range
  // 5. insert allgather_unique_keys to hash table & get allreduce_wgrad table_range
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    HCTR_LIB_THROW(
        cudaMemsetAsync(hash_table.hash_table_.get(), 0, hash_table.hash_table_.nbytes(), stream));
    HCTR_LIB_THROW(cudaMemsetAsync(allreduce_wgrad.table_range.get(), 0,
                                   allreduce_wgrad.table_range.nbytes(), stream));
    dim3 grid_size(num_table, num_gpus, 1);
    constexpr int block_size = 512;
    auto table_ptr = reinterpret_cast<TableEntry<key_t>*>(hash_table.hash_table_.get());
    hash_table_insert_key_and_index_kernel<key_t, DirectHash><<<grid_size, block_size, 0, stream>>>(
        broadcast_result.allgather_unique_keys_.get<key_t>(),
        broadcast_result.reordered_allgather_table_range_.get<uint32_t>(), num_table, num_gpus,
        table_ptr, allreduce_wgrad.table_range.get<uint32_t>());
    size_t temp_nbytes = hash_table.d_temp_scan_table_range_storage_.nbytes();
    cub::DeviceScan::InclusiveSum(hash_table.d_temp_scan_table_range_storage_.get(), temp_nbytes,
                                  allreduce_wgrad.table_range.get<uint32_t>(),
                                  allreduce_wgrad.table_range.get<uint32_t>(), num_table + 1,
                                  stream);
  });
}

void cal_indices_for_sparse_allreduce(const embedding::WgradEvStartIndicesCalculationInput& input,
                                      const BroadcastResult& broadcast_result,
                                      const HashTable& hash_table,
                                      SparseAllreduceCalEVStartIndicesTempStorage& temp_storage,
                                      Wgrad& allreduce_wgrad,
                                      WgradEvStartIndicesCalculationOutput& output, int num_table,
                                      int num_gpus, cudaStream_t stream) {
  auto key_type = input.unique_keys.dtype();
  //   6. select allreduce unique_keys
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    HCTR_LIB_THROW(cudaMemsetAsync(temp_storage.mask_unique_keys_in_allgather_unique_keys_.get(), 0,
                                   temp_storage.mask_unique_keys_in_allgather_unique_keys_.nbytes(),
                                   stream));
    auto table_ptr = reinterpret_cast<TableEntry<key_t>*>(hash_table.hash_table_.get());
    dim3 grid_size(num_table, 1, 1);
    constexpr int block_size = 512;
    hash_table_dump_index_mask_kernel<<<grid_size, block_size, 0, stream>>>(
        table_ptr, broadcast_result.reordered_allgather_table_range_.get<uint32_t>(), num_gpus,
        num_table, temp_storage.mask_unique_keys_in_allgather_unique_keys_.get<int>());
    size_t temp_nbytes = temp_storage.d_temp_select_temp_storage_.nbytes();
    cub::DeviceSelect::Flagged(temp_storage.d_temp_select_temp_storage_.get(), temp_nbytes,
                               broadcast_result.allgather_unique_keys_.get<key_t>(),
                               temp_storage.mask_unique_keys_in_allgather_unique_keys_.get<int>(),
                               allreduce_wgrad.unique_keys.get<key_t>(),
                               allreduce_wgrad.num_unique_keys.get<size_t>(),
                               allreduce_wgrad.unique_keys.get_num_elements(), stream);
  });

  // 7. cal allreduce wgrad table_ids & ev start indices from table_range
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    // not sure if this memset can be removed
    HCTR_LIB_THROW(cudaMemsetAsync(allreduce_wgrad.ev_start_indices.get(), 0,
                                   allreduce_wgrad.ev_start_indices.nbytes(), stream));
    dim3 grid_size(num_table, 1, 1);
    constexpr int block_size = 512;
    table_range_to_table_ids_and_ev_start_indices_kernel<<<grid_size, block_size, 0, stream>>>(
        num_table, allreduce_wgrad.table_range.get<uint32_t>(), input.unique_table_ids.get<int>(),
        allreduce_wgrad.attr.table_id_to_ev_size.get<int>(), allreduce_wgrad.table_ids.get<int>(),
        allreduce_wgrad.ev_start_indices.get<uint32_t>());
    size_t temp_nbytes = temp_storage.d_temp_scan_ev_start_indices_storage_.nbytes();
    cub::DeviceScan::InclusiveSum(temp_storage.d_temp_scan_ev_start_indices_storage_.get(),
                                  temp_nbytes, allreduce_wgrad.ev_start_indices.get<uint32_t>(),
                                  allreduce_wgrad.ev_start_indices.get<uint32_t>(),
                                  allreduce_wgrad.ev_start_indices.get_num_elements(), stream);
  });

  // 8. cal localreduce ev_start_indices
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    size_t temp_nbytes = temp_storage.d_temp_scan_unique_idx_temp_storage_.nbytes();
    cub::DeviceScan::InclusiveSum(
        temp_storage.d_temp_scan_unique_idx_temp_storage_.get(), temp_nbytes,
        temp_storage.mask_unique_keys_in_allgather_unique_keys_.get<int>(),
        temp_storage.unique_idx_.get<int>(), temp_storage.unique_idx_.get_num_elements(), stream);

    auto table_ptr = reinterpret_cast<TableEntry<key_t>*>(hash_table.hash_table_.get());
    dim3 grid_size(num_table, 1, 1);
    constexpr int block_size = 512;
    hash_table_lookup_key_and_map_ev_start_indices<key_t, DirectHash>
        <<<grid_size, block_size, 0, stream>>>(
            input.unique_keys.get<key_t>(), input.table_range.get<uint32_t>(), num_table, num_gpus,
            table_ptr, broadcast_result.reordered_allgather_table_range_.get<uint32_t>(),
            temp_storage.unique_idx_.get<int>(), allreduce_wgrad.table_range.get<uint32_t>(),
            allreduce_wgrad.ev_start_indices.get<uint32_t>(),
            output.ev_start_indices.get<uint32_t>());
  });
}

void SparseAllreduceIndexCalculation::cal_for_sparse_input(const EmbeddingInput& embedding_input,
                                                           ReductionIndices& reduction_indices,
                                                           Wgrad& local_reduce_wgrad,
                                                           Wgrad& allreduce_wgrad, int batch_size) {
  auto sparse_allreduce_cal_ev_start_indices = [&](const WgradEvStartIndicesCalculationInput& input,
                                                   WgradEvStartIndicesCalculationOutput& output,
                                                   cudaStream_t stream) {
    int gpu_id = core_->get_global_gpu_id();
    int num_gpus = core_->get_global_gpu_count();
    auto comm = core_->get_nccl();

    int num_table = input.unique_table_ids.get_num_elements();

    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    broadcast_unique_keys(input, cal_ev_start_indices_storage_.broadcast_result_, num_table,
                          num_gpus, stream, comm);

    unique_broadcast_result_using_hash_table(input, cal_ev_start_indices_storage_.broadcast_result_,
                                             cal_ev_start_indices_storage_.hash_table_,
                                             allreduce_wgrad, num_table, num_gpus, stream);

    cal_indices_for_sparse_allreduce(input, cal_ev_start_indices_storage_.broadcast_result_,
                                     cal_ev_start_indices_storage_.hash_table_,
                                     cal_ev_start_indices_storage_.temp_storage_, allreduce_wgrad,
                                     output, num_table, num_gpus, stream);
  };
  local_reduce_index_calculation_.cal_for_sparse_input(
      embedding_input, segmented_sort_device_, segmented_unique_, cal_dst_ids_, reduction_indices,
      local_reduce_wgrad, batch_size);
  local_reduce_index_calculation_.cal_unique_key_table_range(local_reduce_wgrad);
  local_reduce_index_calculation_.cal_dst_ev_start(local_reduce_wgrad,
                                                   sparse_allreduce_cal_ev_start_indices);
}

}  // namespace embedding
