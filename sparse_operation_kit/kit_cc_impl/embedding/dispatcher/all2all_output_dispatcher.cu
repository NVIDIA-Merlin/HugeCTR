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
#include "common/include/forward_functions.h"

namespace SparseOperationKit {

template <typename EmbeddingType>
__global__ static void
reorderKernel(const size_t EmbeddingDimension,
              EmbeddingType *inputs, uint32_t *indices, EmbeddingType *outputs,
              size_t chunks, size_t max_chunk_size, uint32_t *chunk_sizes) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
       id < chunks * max_chunk_size * EmbeddingDimension;
       id += blockDim.x * gridDim.x) {
    size_t chunk_id = id / (max_chunk_size * EmbeddingDimension);
    size_t row_id = (id - chunk_id * max_chunk_size * EmbeddingDimension) /
                    EmbeddingDimension;
    size_t item_id = id - chunk_id * (max_chunk_size * EmbeddingDimension) -
                     row_id * EmbeddingDimension;

    if (row_id < chunk_sizes[chunk_id]) {
      size_t index =
          static_cast<size_t>(indices[chunk_id * max_chunk_size + row_id]);
      outputs[index * EmbeddingDimension + item_id] = inputs[id];
    }
  }
}

template <typename EmbeddingType>
__global__ static void gatherKernel(const size_t EmbeddingDimension,
                                    EmbeddingType *inputs, uint32_t *indices,
                                    EmbeddingType *outputs, size_t chunks,
                                    size_t max_chunk_size) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
       id < chunks * max_chunk_size * EmbeddingDimension;
       id += blockDim.x * gridDim.x) {
    size_t chunk_id = id / (max_chunk_size * EmbeddingDimension);
    size_t row_id = (id - chunk_id * max_chunk_size * EmbeddingDimension) /
                    EmbeddingDimension;
    size_t item_id = id - chunk_id * (max_chunk_size * EmbeddingDimension) -
                     row_id * EmbeddingDimension;

    size_t index =
        static_cast<size_t>(indices[chunk_id * max_chunk_size + row_id]);
    if (index != static_cast<uint32_t>(-1)) {
      outputs[id] = inputs[index * EmbeddingDimension + item_id];
    }
  }
}


class All2AllOutputDispatcher : public Dispatcher {
public:
    explicit All2AllOutputDispatcher(ConstructionContext_t context)
    : Dispatcher(context), resource_mgr_(base_context()->get_resource_mgr()),
    num_keys_per_rank_(base_context()->get_replica_batch_size() * 
                       base_context()->get_slot_num() * 
                       base_context()->get_nnz_per_slot()) {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        exchanged_embeddings_buf_.reserve(local_gpu_count);
        gathered_gradients_buf_.reserve(local_gpu_count);
    }

    void allocate_forward_spaces() override {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
            auto &buffer = base_context()->get_buffer(dev_id);
            {
                Tensor2<float> tensor;
                buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
                exchanged_embeddings_buf_.push_back(tensor);
            }
        } // for dev_id in local_gpu_count
    }

    void allocate_backward_spaces() override {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
            auto &buffer = base_context()->get_buffer(dev_id);

            {
                Tensor2<float> tensor;
                buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
                gathered_gradients_buf_.push_back(tensor);
            }
        } // for dev_id in local_gpu_count
    }

    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        const auto &replica_gathered_embeddings = replica_context->input("replica_gathered_embeddings");
        const auto &h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
        const auto &h_num_exchanged_keys = replica_context->input("replica_h_num_exchanged_keys");
        const auto &h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");
        const auto &replica_num_selected_keys = replica_context->input("replica_num_selected_keys");
        const auto &replica_selected_indices_buf = replica_context->input("replica_selected_indices_buf");

        auto &replica_output = replica_context->output("replica_output");
        // step 1: exchange embedding values among all GPUs.
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
            CK_NCCL(ncclSend(replica_gathered_embeddings->GetPtrWithType<float>() + 
                             h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             h_num_exchanged_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             ncclFloat32, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
            CK_NCCL(ncclRecv(exchanged_embeddings_buf_[local_replica_id].get_ptr() +
                             dev_id * num_keys_per_rank_ * embedding_vec_size,
                             h_num_selected_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             ncclFloat32, /*peer=*/dev_id,
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
        } // for dev_id in global_gpu_count
        CK_NCCL(ncclGroupEnd());

        // step 2: reorder embedding values
        reorderKernel<float>
            <<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
                embedding_vec_size,
                exchanged_embeddings_buf_[local_replica_id].get_ptr(), 
                replica_selected_indices_buf->GetPtrWithType<uint32_t>(),
                replica_output->GetPtrWithType<float>(), 
                /*chunks=*/global_gpu_count, 
                num_keys_per_rank_, 
                replica_num_selected_keys->GetPtrWithType<uint32_t>());

    }
    void backward(const Context_t &replica_context) override {
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        const auto &replica_top_gradients = replica_context->input("replica_top_gradient");
        const auto &replica_selected_indices_buf = replica_context->input("replica_selected_indices_buf");
        const auto &replica_h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
        const auto &h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");
        const auto &h_num_exchanged_keys = replica_context->input("replica_h_num_exchanged_keys");

        auto &replica_input_grad = replica_context->output("replica_input_grad");

        // step 1: gather top gradients for local GPU.
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        gatherKernel<<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
            embedding_vec_size,
            replica_top_gradients->GetPtrWithType<float>(),
            replica_selected_indices_buf->GetPtrWithType<uint32_t>(),
            gathered_gradients_buf_[local_replica_id].get_ptr(),
            /*chunks=*/global_gpu_count, 
            num_keys_per_rank_);

        // step 2: exchange gradients among all GPUs.
        CK_NCCL(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
            CK_NCCL(ncclSend(gathered_gradients_buf_[local_replica_id].get_ptr() + 
                                dev_id * num_keys_per_rank_ * embedding_vec_size,
                             h_num_selected_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             ncclFloat32, /*peer=*/dev_id, 
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
            CK_NCCL(ncclRecv(replica_input_grad->GetPtrWithType<float>() + 
                                replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             h_num_exchanged_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                             ncclFloat32, /*peer=*/dev_id,
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
        } // for dev_id in global_gpu_count
        CK_NCCL(ncclGroupEnd());
    }   
private:
    std::shared_ptr<ResourcesManager> resource_mgr_;
    const size_t num_keys_per_rank_;

    // forward spaces
    Tensors2<float> exchanged_embeddings_buf_;

    // backward spaces
    Tensors2<float> gathered_gradients_buf_;
};

REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", All2AllOutputDispatcher);

} // namespace SparseOperationKit