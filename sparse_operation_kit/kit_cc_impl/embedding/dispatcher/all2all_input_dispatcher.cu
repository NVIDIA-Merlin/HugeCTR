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

template <typename Key> struct IdenticalHash {
  using argument_type = Key;
  using result_type = uint32_t;

  IdenticalHash() = default;

  template <typename TKey>
  static __device__ result_type compute(TKey const &key) {
    return static_cast<result_type>(key);
  }
};


/*It will dispatcher keys based on key % GPU_NUM */
template <typename KeyType, typename Hasher>
__global__ static void
selectKernel(KeyType *input_keys, size_t num_keys, KeyType *output_keys,
             uint32_t *output_indices, size_t chunks, size_t max_chunk_size,
             uint32_t *chunk_sizes) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_keys;
       id += blockDim.x * gridDim.x) {
    KeyType key = input_keys[id];
    size_t chunk_id = Hasher::compute(key) % chunks;

    uint32_t offset = atomicAdd(chunk_sizes + chunk_id, 1);
    output_keys[chunk_id * max_chunk_size + offset] = key;
    output_indices[chunk_id * max_chunk_size + offset] = id;
  }
}



class All2AllInputDispatcher : public Dispatcher {
public:
    explicit All2AllInputDispatcher(ConstructionContext_t context) 
    : Dispatcher(context), resource_mgr_(base_context()->get_resource_mgr()),
    num_keys_per_rank_(base_context()->get_replica_batch_size() * 
                       base_context()->get_slot_num() * 
                       base_context()->get_nnz_per_slot()) {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        selected_keys_buf_.reserve(local_gpu_count);
        selected_indices_buf_.reserve(local_gpu_count);
        num_selected_keys_.reserve(local_gpu_count);
        num_exchanged_keys_.reserve(local_gpu_count);
        h_num_selected_keys_.reserve(local_gpu_count);
        h_num_exchanged_keys_.reserve(local_gpu_count);
        exchanged_keys_buf_.reserve(local_gpu_count);
        h_recv_chunk_offsets_.reserve(local_gpu_count);
        h_send_chunk_offsets_.reserve(local_gpu_count);
    }

    void allocate_forward_spaces() override {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
            auto &buffer = base_context()->get_buffer(dev_id);
            auto &host_buffer = base_context()->get_host_buffer(dev_id);

            {
                Tensor2<int64_t> tensor;
                buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
                selected_keys_buf_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
                selected_indices_buf_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                buffer->reserve({global_gpu_count}, &tensor);
                num_selected_keys_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                buffer->reserve({global_gpu_count}, &tensor);
                num_exchanged_keys_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                host_buffer->reserve({global_gpu_count}, &tensor);
                h_num_selected_keys_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                host_buffer->reserve({global_gpu_count}, &tensor);
                h_num_exchanged_keys_.push_back(tensor);
            }
            {
                Tensor2<int64_t> tensor;
                buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
                exchanged_keys_buf_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                host_buffer->reserve({global_gpu_count + 1}, &tensor);
                h_recv_chunk_offsets_.push_back(tensor);
            }
            {
                Tensor2<uint32_t> tensor;
                host_buffer->reserve({global_gpu_count + 1}, &tensor);
                h_send_chunk_offsets_.push_back(tensor);
            }
        } // for dev_id in local_gpu_count
    }

    void allocate_backward_spaces() override {}

    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        // step 1: reset count spaces.
        CK_CUDA(cudaMemsetAsync(num_selected_keys_[local_replica_id].get_ptr(), 0, 
                                num_selected_keys_[local_replica_id].get_size_in_bytes(), 
                                local_gpu->get_stream()));
        CK_CUDA(cudaMemsetAsync(selected_indices_buf_[local_replica_id].get_ptr(), -1,
                                selected_indices_buf_[local_replica_id].get_size_in_bytes(),
                                local_gpu->get_stream()));
        // FIXME: should use cudaMemcpyAsync??
        // will std::memset be optimized away??
        std::memset(h_recv_chunk_offsets_[local_replica_id].get_ptr(), 0, 
                    h_recv_chunk_offsets_[local_replica_id].get_size_in_bytes());
        std::memset(h_send_chunk_offsets_[local_replica_id].get_ptr(), 0,
                    h_send_chunk_offsets_[local_replica_id].get_size_in_bytes());

        // step 2: select keys for each GPU (rank)
        const auto &input_keys = replica_context->input("replica_values");
        selectKernel<int64_t, IdenticalHash<int64_t>>
            <<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
                input_keys->GetPtrWithType<int64_t>(), input_keys->get_num_elements(),
                selected_keys_buf_[local_replica_id].get_ptr(), 
                selected_indices_buf_[local_replica_id].get_ptr(),
                /*chunks=*/global_gpu_count, num_keys_per_rank_, 
                num_selected_keys_[local_replica_id].get_ptr());
        CK_CUDA(cudaGetLastError());

        // step 3: exchange selected keys count among all GPUs
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
            CK_NCCL(ncclSend(num_selected_keys_[local_replica_id].get_ptr() + dev_id, 1, 
                             ncclUint32, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
            CK_NCCL(ncclRecv(num_exchanged_keys_[local_replica_id].get_ptr() + dev_id, 1,
                             ncclUint32, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
        } // for dev_id in global_gpu_count
        CK_NCCL(ncclGroupEnd());

        // step 4: copy count from GPU to CPU and calculate count offsets
        CK_CUDA(cudaMemcpyAsync(h_num_selected_keys_[local_replica_id].get_ptr(), 
                                num_selected_keys_[local_replica_id].get_ptr(),
                                num_selected_keys_[local_replica_id].get_size_in_bytes(),
                                cudaMemcpyDeviceToHost, local_gpu->get_stream()));
        CK_CUDA(cudaMemcpyAsync(h_num_exchanged_keys_[local_replica_id].get_ptr(),
                                num_exchanged_keys_[local_replica_id].get_ptr(),
                                num_exchanged_keys_[local_replica_id].get_size_in_bytes(),
                                cudaMemcpyDeviceToHost, local_gpu->get_stream()));
        resource_mgr_->sync_gpu(local_replica_id);
        for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
            h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id + 1] =
                h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id] + 
                h_num_exchanged_keys_[local_replica_id].get_ptr()[dev_id];

            h_send_chunk_offsets_[local_replica_id].get_ptr()[dev_id + 1] = 
                h_send_chunk_offsets_[local_replica_id].get_ptr()[dev_id] +
                h_num_selected_keys_[local_replica_id].get_ptr()[dev_id];
        } // for dev_id in global_gpu_count

        // step 5: exchange selected keys among all GPUs
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
            CK_NCCL(ncclSend(selected_keys_buf_[local_replica_id].get_ptr() + dev_id * num_keys_per_rank_,
                             h_num_selected_keys_[local_replica_id].get_ptr()[dev_id],
                             ncclInt64, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
            CK_NCCL(ncclRecv(exchanged_keys_buf_[local_replica_id].get_ptr() + h_recv_chunk_offsets_[local_replica_id].get_ptr()[dev_id],
                             h_num_exchanged_keys_[local_replica_id].get_ptr()[dev_id],
                             ncclInt64, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
        } // for dev_id in global_gpu_count
        CK_NCCL(ncclGroupEnd());

        // set output of this dispatcher
        replica_context->set_output("replica_exchanged_keys", exchanged_keys_buf_[local_replica_id]);
        replica_context->set_output("replica_h_recv_chunk_offsets", h_recv_chunk_offsets_[local_replica_id]);
        replica_context->set_output("replica_h_send_chunk_offsets", h_send_chunk_offsets_[local_replica_id]);
        replica_context->set_output("replica_h_num_exchanged_keys", h_num_exchanged_keys_[local_replica_id]);
        replica_context->set_output("replica_h_num_selected_keys", h_num_selected_keys_[local_replica_id]);
        replica_context->set_output("replica_num_selected_keys", num_selected_keys_[local_replica_id]);
        replica_context->set_output("replica_selected_indices_buf", selected_indices_buf_[local_replica_id]);
    }
    void backward(const Context_t &replica_context) override {}

private:
    const std::shared_ptr<ResourcesManager> resource_mgr_;
    const size_t num_keys_per_rank_;

    // forward spaces
    Tensors2<int64_t> selected_keys_buf_;
    Tensors2<uint32_t> selected_indices_buf_;
    Tensors2<uint32_t> num_selected_keys_;
    Tensors2<uint32_t> num_exchanged_keys_;
    Tensors2<uint32_t> h_num_selected_keys_;
    Tensors2<uint32_t> h_num_exchanged_keys_;
    Tensors2<int64_t> exchanged_keys_buf_;
    Tensors2<uint32_t> h_recv_chunk_offsets_;
    Tensors2<uint32_t> h_send_chunk_offsets_;
};

REGISTER_INPUT_DISPATCHER_BUILDER("All2AllInput", All2AllInputDispatcher);

}  // namespace SparseOperationKit
