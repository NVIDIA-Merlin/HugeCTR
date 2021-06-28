/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 
#include "operation/operation_interface.h"
#include <cub/cub.cuh>

namespace SparseOperationKit {

template <typename KeyType, typename Hasher>
struct SelectOp {
  size_t rank_;
  size_t ranks_;

  __device__ __host__ __forceinline__ SelectOp(size_t rank, size_t ranks)
      : rank_(rank), ranks_(ranks) {}

  __device__ __forceinline__ bool operator()(KeyType key) {
    size_t target_rank = Hasher::compute(key) % ranks_;
    return target_rank == rank_;
  }
};

template <typename Key, uint32_t m_seed = 0>
struct MurmurHash3_32 {
  using argument_type = Key;
  using result_type = uint32_t;

  MurmurHash3_32() = default;

  static __device__ uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }

  static __device__ uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  template <typename TKey>
  static __device__ result_type compute(TKey const &key) {
    constexpr int len = sizeof(argument_type);
    uint8_t const *const data = reinterpret_cast<uint8_t const *>(&key);
    constexpr int nblocks = len / 4;

    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    uint32_t const *const blocks = reinterpret_cast<uint32_t const *>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];  // getblock32(blocks,i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    uint8_t const *tail = reinterpret_cast<uint8_t const *>(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
      case 3:
        k1 ^= tail[2] << 16;
      case 2:
        k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }
};

template <typename T>
inline ncclDataType_t getNcclDataType();

template <>
inline ncclDataType_t getNcclDataType<int64_t>() {
  return ncclInt64;
}
template <>
inline ncclDataType_t getNcclDataType<size_t>() {
  return ncclUint64;
}
template <>
inline ncclDataType_t getNcclDataType<float>() {
  return ncclFloat32;
}

template <typename KeyType>
__global__ static void aggregateKernel(KeyType *input_keys, KeyType *output_keys, size_t chunks,
                                       size_t max_chunk_size, size_t chunk1_offset,
                                       size_t chunk2_offset, size_t chunk3_offset,
                                       size_t chunk4_offset, size_t chunk5_offset,
                                       size_t chunk6_offset, size_t chunk7_offset,
                                       size_t next_chunk_offset) {
  const size_t chunk_offsets[] = {0,
                                  chunk1_offset,
                                  chunk2_offset,
                                  chunk3_offset,
                                  chunk4_offset,
                                  chunk5_offset,
                                  chunk6_offset,
                                  chunk7_offset,
                                  next_chunk_offset};

  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < chunks * max_chunk_size;
       id += blockDim.x * gridDim.x) {
    size_t chunk_id = id / max_chunk_size;
    size_t item_id = id - chunk_id * max_chunk_size;

    if (item_id < chunk_offsets[chunk_id + 1] - chunk_offsets[chunk_id]) {
      output_keys[chunk_offsets[chunk_id] + item_id] = input_keys[id];
    }
  }
}


class All2AllInputDispatcher : public Dispatcher {
 public:
  All2AllInputDispatcher(ConstructionContext_t context)
      : Dispatcher(context),
        resouces_mgr_(base_context()->get_resource_mgr()),
        num_keys_per_rank_(base_context()->get_nnz_per_slot()) {
    const size_t local_gpu_count = resouces_mgr_->get_local_gpu_count();
    cub_temp_storage_.reserve(local_gpu_count);
    selected_keys_buf_.reserve(local_gpu_count);
    num_selected_keys_.reserve(local_gpu_count);
    num_gathered_keys_.reserve(local_gpu_count);
    h_num_selected_keys_.reserve(local_gpu_count);
    h_num_gathered_keys_.reserve(local_gpu_count);
    h_recv_chunk_offsets_.reserve(local_gpu_count);
    h_send_chunk_offsets_.reserve(local_gpu_count);
    gathered_keys_buf_.reserve(local_gpu_count);
    aggregated_keys_buf_.reserve(local_gpu_count);
  }

  void allocate_forward_spaces(const size_t global_batch_size) override {
    const size_t local_gpu_count = resouces_mgr_->get_local_gpu_count();
    const size_t replica_batch_size = global_batch_size / resouces_mgr_->get_global_gpu_count();

    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_buffer(dev_id);
      {
        size_t size = 0;
        SelectOp<int64_t, MurmurHash3_32<int64_t>> select_op(0,
                                                             resouces_mgr_->get_global_gpu_count());
        CK_CUDA(cub::DeviceSelect::If(
            nullptr, size, static_cast<int64_t *>(nullptr), static_cast<int64_t *>(nullptr),
            static_cast<size_t *>(nullptr), num_keys_per_rank_, select_op));

        Tensor2<void> tensor;
        buffer->reserve({size}, &tensor);
        cub_temp_storage_.push_back(tensor);
      }
      {
        Tensor2<int64_t> tensor;
        buffer->reserve({num_keys_per_rank_ * local_gpu_count}, &tensor);
        selected_keys_buf_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        buffer->reserve({local_gpu_count}, &tensor);
        num_selected_keys_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        buffer->reserve({local_gpu_count}, &tensor);
        num_gathered_keys_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        host_buffer->reserve({local_gpu_count}, &tensor);
        h_num_selected_keys_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        host_buffer->reserve({local_gpu_count}, &tensor);
        h_num_gathered_keys_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        host_buffer->reserve({local_gpu_count + 1}, &tensor);
        h_recv_chunk_offsets_.push_back(tensor);
      }
      {
        Tensor2<size_t> tensor;
        host_buffer->reserve({local_gpu_count + 1}, &tensor);
        h_send_chunk_offsets_.push_back(tensor);
      }
      {
        Tensor2<int64_t> tensor;
        buffer->reserve({num_keys_per_rank_ * local_gpu_count}, &tensor);
        gathered_keys_buf_.push_back(tensor);
      }
      {
        Tensor2<int64_t> tensor;
        buffer->reserve({num_keys_per_rank_ * local_gpu_count}, &tensor);
        aggregated_keys_buf_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }
  void allocate_backward_spaces(const size_t global_batch_size) override {}

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t local_gpu_count = resouces_mgr_->get_local_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resouces_mgr_->cal_local_id_from_global_id(global_replica_id);

    const auto &local_gpu = resouces_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_values = replica_context->input("replica_values");

    for (size_t i = 0; i < local_gpu_count; i++) {
      SelectOp<int64_t, MurmurHash3_32<int64_t>> select_op(i, local_gpu_count);
      size_t size = cub_temp_storage_[local_replica_id].get_size_in_bytes();
      CK_CUDA(cub::DeviceSelect::If(
          cub_temp_storage_[local_replica_id].get_ptr(), size,
          replica_values->GetPtrWithType<long long>(),
          selected_keys_buf_[local_replica_id].get_ptr() + i * num_keys_per_rank_,
          num_selected_keys_[local_replica_id].get_ptr() + i, replica_values->get_num_elements(),
          select_op, local_gpu->get_stream()));
    }  // for r in local_gpu_count

    CK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < local_gpu_count; i++) {
      CK_NCCL(ncclSend(num_selected_keys_[local_replica_id].get_ptr() + i, 1,
                       getNcclDataType<size_t>(), i, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
      CK_NCCL(ncclRecv(num_gathered_keys_[local_replica_id].get_ptr() + i, 1,
                       getNcclDataType<size_t>(), i, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
    }
    CK_NCCL(ncclGroupEnd());

    CK_CUDA(cudaMemcpyAsync(h_num_selected_keys_[local_replica_id].get_ptr(),
                            num_selected_keys_[local_replica_id].get_ptr(),
                            sizeof(size_t) * local_gpu_count, cudaMemcpyDeviceToHost,
                            local_gpu->get_stream()));
    CK_CUDA(cudaMemcpyAsync(h_num_gathered_keys_[local_replica_id].get_ptr(),
                            num_gathered_keys_[local_replica_id].get_ptr(),
                            sizeof(size_t) * local_gpu_count, cudaMemcpyDeviceToHost,
                            local_gpu->get_stream()));
    resouces_mgr_->sync_gpu(local_replica_id);

    CK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < local_gpu_count; i++) {
      CK_NCCL(ncclSend(selected_keys_buf_[local_replica_id].get_ptr() + i * num_keys_per_rank_,
                       h_num_selected_keys_[local_replica_id].get_ptr()[i],
                       getNcclDataType<int64_t>(), i, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
      CK_NCCL(ncclRecv(gathered_keys_buf_[local_replica_id].get_ptr() + i * num_keys_per_rank_,
                       h_num_gathered_keys_[local_replica_id].get_ptr()[i],
                       getNcclDataType<int64_t>(), i, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
    }
    CK_NCCL(ncclGroupEnd());

    try {
        std::memset(h_recv_chunk_offsets_[local_replica_id].get_ptr(), 0, 
                    h_recv_chunk_offsets_[local_replica_id].get_size_in_bytes());
        for (size_t r = 0; r < local_gpu_count; r++) {
          h_recv_chunk_offsets_[local_replica_id].get_ptr()[r + 1] = 
                h_recv_chunk_offsets_[local_replica_id].get_ptr()[r] 
                + h_num_gathered_keys_[local_replica_id].get_ptr()[r];
        }
        std::memset(h_send_chunk_offsets_[local_replica_id].get_ptr(), 0,
                    h_send_chunk_offsets_[local_replica_id].get_size_in_bytes());
        for (size_t r = 0; r < local_gpu_count; r++) {
            h_send_chunk_offsets_[local_replica_id].get_ptr()[r + 1] =
                h_send_chunk_offsets_[local_replica_id].get_ptr()[r]
                + h_num_selected_keys_[local_replica_id].get_ptr()[r];
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }

    aggregateKernel<int64_t><<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
        gathered_keys_buf_[local_replica_id].get_ptr(),
        aggregated_keys_buf_[local_replica_id].get_ptr(), local_gpu_count, num_keys_per_rank_,
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[1], 
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[2], 
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[3],
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[4], 
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[5], 
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[6],
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[7], 
        h_recv_chunk_offsets_[local_replica_id].get_ptr()[8]);
    CK_CUDA(cudaGetLastError());

    // set output
    replica_context->set_output("replica_aggregated_keys", aggregated_keys_buf_[local_replica_id]);
    replica_context->set_output("replica_h_recv_chunk_offsets", h_recv_chunk_offsets_[local_replica_id]);
    replica_context->set_output("replica_h_send_chunk_offsets", h_send_chunk_offsets_[local_replica_id]);
    replica_context->set_output("replica_h_num_gathered_keys", h_num_gathered_keys_[local_replica_id]);
    replica_context->set_output("replica_h_num_selected_keys", h_num_selected_keys_[local_replica_id]);
  }

  void backward(const Context_t &replica_context) override {
      
  }

 private:
  std::shared_ptr<ResourcesManager> resouces_mgr_;
  // FIXME: what does this attribute mean? it should be a larger number?
  // FIXME: it is also used in dense_gather.cu
  const size_t num_keys_per_rank_; 

  Tensors2<void> cub_temp_storage_;
  Tensors2<int64_t> selected_keys_buf_;
  Tensors2<size_t> num_selected_keys_;
  Tensors2<size_t> num_gathered_keys_;
  Tensors2<size_t> h_num_selected_keys_;
  Tensors2<size_t> h_num_gathered_keys_;
  Tensors2<size_t> h_recv_chunk_offsets_;
  Tensors2<size_t> h_send_chunk_offsets_;
  Tensors2<int64_t> gathered_keys_buf_;
  Tensors2<int64_t> aggregated_keys_buf_;
};

REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", All2AllInputDispatcher);

}  // namespace SparseOperationKit
