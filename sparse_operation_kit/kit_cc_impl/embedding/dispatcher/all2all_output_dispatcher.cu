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

namespace SparseOperationKit {

class All2AllOutputDispatcher : public Dispatcher {
public:
    explicit All2AllOutputDispatcher(ConstructionContext_t context)
    : Dispatcher(context),
    resource_mgr_(base_context()->get_resource_mgr()),
    embedding_vec_size_(base_context()->get_param()->get_embedding_vec_size())
    {}

    void allocate_forward_spaces(const size_t global_batch_size) override {}

    void allocate_backward_spaces(const size_t global_batch_size) override {}

    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto& local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();

        const auto &replica_h_num_gathered_keys = replica_context->input("replica_h_num_gathered_keys");
        const auto &replica_h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
        const auto &replica_h_send_chunk_offsets = replica_context->input("replica_h_send_chunk_offsets");
        const auto &gathered_embeddings_buf = replica_context->input("gathered_embeddings_buf");
        const auto &replica_h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");

        CK_NCCL(ncclGroupStart());
        for (size_t i = 0; i < local_gpu_count; i++) {
        CK_NCCL(ncclSend(
            gathered_embeddings_buf->GetPtrWithType<float>() +
                replica_h_recv_chunk_offsets->GetPtrWithType<size_t>()[i] * embedding_vec_size_,
            replica_h_num_gathered_keys->GetPtrWithType<size_t>()[i] * embedding_vec_size_,
            ncclFloat32, i, local_gpu->get_nccl(), local_gpu->get_stream()));
        CK_NCCL(ncclRecv(
            replica_context->output("replica_output")->GetPtrWithType<float>() +
                replica_h_send_chunk_offsets->GetPtrWithType<size_t>()[i] * embedding_vec_size_,
            replica_h_num_selected_keys->GetPtrWithType<size_t>()[i] * embedding_vec_size_,
            ncclFloat32, i, local_gpu->get_nccl(), local_gpu->get_stream()));
        }
        CK_NCCL(ncclGroupEnd());
    }

    void backward(const Context_t &replica_context) override {}

private:
    std::shared_ptr<ResourcesManager> resource_mgr_;

    const size_t embedding_vec_size_;
};

REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", All2AllOutputDispatcher);

} // namespace SparseOperationKit