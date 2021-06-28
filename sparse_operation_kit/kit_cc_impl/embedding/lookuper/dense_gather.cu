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

template <typename EmbeddingType>
__global__ static void gatherKernel(size_t embedding_dimension,
                                    const EmbeddingType *input_embeddings, size_t *indices,
                                    size_t num_indices, EmbeddingType *output_embeddings) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_indices * embedding_dimension;
       id += blockDim.x * gridDim.x) {
    size_t item_id = id / embedding_dimension;
    size_t embedding_id = id - item_id * embedding_dimension;

    output_embeddings[id] = input_embeddings[indices[item_id] * embedding_dimension + embedding_id];
  }
}

class DenseGather : public EmbeddingLookuper {
public:
    DenseGather(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
    : EmbeddingLookuper(context, param),
    resource_mgr_(base_context()->get_resource_mgr()),
    num_keys_per_rank_(base_context()->get_nnz_per_slot())
    {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        mapped_indices_buf_.reserve(local_gpu_count);
        gathered_embeddings_buf_.reserve(local_gpu_count);
    }

    void allocate_forward_spaces(size_t const global_batch_size) override {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
            auto &buffer = base_context()->get_buffer(dev_id);
            {
                Tensor2<size_t> tensor;
                buffer->reserve({num_keys_per_rank_ * local_gpu_count}, &tensor);
                mapped_indices_buf_.push_back(tensor);
            }
            {
                Tensor2<float> tensor;
                buffer->reserve({base_context()->get_param()->get_embedding_vec_size(), 
                                 local_gpu_count * num_keys_per_rank_}, &tensor);
                gathered_embeddings_buf_.push_back(tensor);
            }
        } // for dev_id in local_gpu_count
    }

    void allocate_backward_spaces(size_t const global_batch_size) override {}

    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();

        // get input tensor
        const auto &replica_aggregated_keys = replica_context->input("replica_aggregated_keys");
        const auto &replica_h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");

        // get hash_value_index from hash_table by hash_key
        auto &hashtable = param_->get_hashtable(local_replica_id);
        if (!hashtable) throw std::runtime_error(ErrorBase + "No available hashtable.");
        if (training) {
            hashtable->get_insert(replica_aggregated_keys->GetPtrWithType<int64_t>(),
                                  mapped_indices_buf_[local_replica_id].get_ptr(),
                                  replica_h_recv_chunk_offsets->GetPtrWithType<size_t>()[local_gpu_count],
                                  local_gpu->get_stream());
        } else {
            hashtable->get(replica_aggregated_keys->GetPtrWithType<int64_t>(),
                           mapped_indices_buf_[local_replica_id].get_ptr(),
                           replica_h_recv_chunk_offsets->GetPtrWithType<size_t>()[local_gpu_count],
                           local_gpu->get_stream());
        }

        gatherKernel<float><<<local_gpu->get_sm_count() * 2, 1024, 0, local_gpu->get_stream()>>>(
            param_->get_embedding_vec_size(), 
            param_->get_embedding_table_tensor(local_replica_id)->GetPtrWithType<float>(),
            mapped_indices_buf_[local_replica_id].get_ptr(), 
            replica_h_recv_chunk_offsets->GetPtrWithType<size_t>()[local_gpu_count],
            gathered_embeddings_buf_[local_replica_id].get_ptr());
        CK_CUDA(cudaGetLastError());

        // set output
        replica_context->set_output("gathered_embeddings_buf", gathered_embeddings_buf_[local_replica_id]);

        
    }

    void backward(const Context_t &replica_context) override {}

    void load_tensors_to_memory(const std::vector<std::shared_ptr<Tensor>>& tensors) override {}

private:
    std::shared_ptr<ResourcesManager> resource_mgr_;
    const size_t num_keys_per_rank_;

    Tensors2<size_t> mapped_indices_buf_;
    Tensors2<float> gathered_embeddings_buf_;
};

REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DenseGather);

} // namespace SparseOperationKit