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
__global__ static void gatherKernel(const size_t EmbeddingDimension,
                                    EmbeddingType *inputs, size_t *indices, 
                                    size_t num_indices, EmbeddingType *outputs) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
       id < num_indices * EmbeddingDimension; id += blockDim.x * gridDim.x) {

    size_t item_id = id / EmbeddingDimension;
    size_t embedding_id = id - item_id * EmbeddingDimension;

    size_t index = static_cast<size_t>(indices[item_id]);
    outputs[id] = inputs[index * EmbeddingDimension + embedding_id];
  }
}


class DenseGather : public EmbeddingLookuper {
public:
    DenseGather(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
    : EmbeddingLookuper(context, param),
    resource_mgr_(base_context()->get_resource_mgr()),
    num_keys_per_rank_(base_context()->get_replica_batch_size() * 
                       base_context()->get_slot_num() * 
                       base_context()->get_nnz_per_slot()) {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        mapped_indices_buf_.reserve(local_gpu_count);
        host_nnz_.reserve(local_gpu_count);
        gathered_embeddings_buf_.reserve(local_gpu_count);

        if (sizeof(size_t) != sizeof(int64_t))
            throw std::runtime_error("In this platform, sizeof(size_t) != sizeof(int64_t). "
                                     "It will cause unexpected behavoir when copy datas from "
                                     "size_t pointer to int64_t pointer.");
    }

    void allocate_forward_spaces() override {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
            auto &buffer = base_context()->get_buffer(dev_id);
            auto &host_buffer = base_context()->get_host_buffer(dev_id);
            {
                Tensor2<size_t> tensor;
                buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
                mapped_indices_buf_.push_back(tensor);
            }
            {
                Tensor2<float> tensor;
                buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
                gathered_embeddings_buf_.push_back(tensor);
            }
            {
                Tensor2<size_t> tensor;
                host_buffer->reserve({1}, &tensor);
                host_nnz_.push_back(tensor);
            }
        } // for dev_id in local_gpu_count
    }
    void allocate_backward_spaces() override {}
    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        auto &hashtable = param_->get_hashtable(local_replica_id);
        if (!hashtable) throw std::runtime_error(ErrorBase + "No avaiable hashtable.");

        const auto &replica_exchanged_keys = replica_context->input("replica_exchanged_keys");
        const auto &replica_h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
        // step 1: get index using keys
        if (training) {
            hashtable->get_insert(replica_exchanged_keys->GetPtrWithType<int64_t>(),
                                  mapped_indices_buf_[local_replica_id].get_ptr(),
                                  /*nnz=*/replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count],
                                  local_gpu->get_stream());
        } else {
            hashtable->get(replica_exchanged_keys->GetPtrWithType<int64_t>(),
                           mapped_indices_buf_[local_replica_id].get_ptr(),
                           /*nnz=*/replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count],
                           local_gpu->get_stream());
        }

        // step 2: gather embedding vectors from embedding table
        const auto &embedding_table = param_->get_embedding_table_tensor(local_replica_id);
        gatherKernel<float><<<local_gpu->get_sm_count() * 2, 1024ul, 0, local_gpu->get_stream()>>>(
            /*EmbeddingDimension=*/param_->get_embedding_vec_size(),
            /*inputs=*/embedding_table->GetPtrWithType<float>(), 
            /*indices=*/mapped_indices_buf_[local_replica_id].get_ptr(),
            /*num_indices=*/replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count],
            /*outputs=*/gathered_embeddings_buf_[local_replica_id].get_ptr());
        CK_CUDA(cudaGetLastError());

        // step 3: set the output of embedding lookuper
        replica_context->set_output("replica_gathered_embeddings", gathered_embeddings_buf_[local_replica_id]);
        // write host_nnz in current iteration
        host_nnz_[local_replica_id].get_ptr()[0] = static_cast<size_t>(
            replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count]);
        replica_context->set_output("replica_host_nnz", host_nnz_[local_replica_id]);
    }
    
    void backward(const Context_t &replica_context) override {
        const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        const auto &replica_h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
        auto &replica_value_index_tensor = replica_context->output("value_index_tensor");

        // FIXME: what if sizeof(size_t) != sizeof(int64_t)
        CK_CUDA(cudaMemcpyAsync(replica_value_index_tensor->GetPtrWithType<int64_t>(),
                                mapped_indices_buf_[local_replica_id].get_ptr(),
                                sizeof(size_t) * replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count],
                                cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
    }
    void restore_params(const std::shared_ptr<Tensor> &keys,
                        const std::shared_ptr<Tensor> &embedding_values,
                        const size_t num_total_keys) override {
        // this lookuper distribute keys to each GPU based on key % GPU_NUM
        const size_t total_max_vocabulary_size = 
            param_->get_max_vocabulary_size_per_gpu() * resource_mgr_->get_global_gpu_count();
        
        MESSAGE("num_total_keys = " + std::to_string(num_total_keys) + ", "
                "while total_max_vocabulary_size = " + std::to_string(total_max_vocabulary_size));

        const int64_t *key_ptr = keys->GetPtrWithType<int64_t>();
        const float *embedding_ptr = embedding_values->GetPtrWithType<float>();

        // step 1: allocate temporary spaces
        const size_t worker_id = resource_mgr_->get_worker_id();
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        constexpr size_t chunk_size = 1000;
        constexpr size_t hash_table_key_tile_size = 1;
        const size_t embedding_vec_size = param_->get_embedding_vec_size();
        const size_t hash_table_key_tile_size_in_bytes = hash_table_key_tile_size * sizeof(int64_t);
        const size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_size;
        const size_t hash_table_key_chunk_size_in_bytes = hash_table_key_chunk_size * sizeof(int64_t);
        const size_t hash_table_value_index_chunk_size_in_bytes = hash_table_key_chunk_size * sizeof(size_t);
        const size_t hash_table_value_tile_size = embedding_vec_size;
        const size_t hash_table_value_tile_size_in_bytes = hash_table_value_tile_size * sizeof(float);
        const size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
        const size_t hash_table_value_chunk_size_in_bytes = hash_table_value_chunk_size * sizeof(float);

        // cannot decide precise the number of values for each GPU, so allocate enough spaces
        std::unique_ptr<size_t []> tile_counter_per_gpu(new size_t[local_gpu_count]());
        std::unique_ptr<size_t []> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]());
        std::unique_ptr<size_t *[]> d_hash_table_value_index_chunk_per_gpu(new size_t *[local_gpu_count]);
        std::unique_ptr<int64_t *[]> h_hash_table_key_chunk_per_gpu(new int64_t *[local_gpu_count]);
        std::unique_ptr<int64_t *[]> d_hash_table_key_chunk_per_gpu(new int64_t *[local_gpu_count]);
        std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);

        HugeCTR::CudaDeviceContext context;
        for (size_t id = 0; id < local_gpu_count; id++) {
            const auto& local_gpu = resource_mgr_->get_local_gpu(id);
            context.set_device(local_gpu->get_local_device_id());

            CK_CUDA(cudaMalloc(&d_hash_table_value_index_chunk_per_gpu[id], 
                                hash_table_value_index_chunk_size_in_bytes));
            CK_CUDA(cudaMemsetAsync(d_hash_table_value_index_chunk_per_gpu[id], 0,
                                    hash_table_value_index_chunk_size_in_bytes,
                                    local_gpu->get_stream()));
            CK_CUDA(cudaMallocHost(&h_hash_table_key_chunk_per_gpu[id],
                                    hash_table_key_chunk_size_in_bytes));
            CK_CUDA(cudaMalloc(&d_hash_table_key_chunk_per_gpu[id], 
                                hash_table_key_chunk_size_in_bytes));
            CK_CUDA(cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], 
                                    hash_table_value_chunk_size_in_bytes));
        } // for id in local_gpu_count
        resource_mgr_->sync_local_gpus();

        // step 2: do uploading
        size_t loop_num = num_total_keys / chunk_size;
        MESSAGE("Worker " + std::to_string(worker_id) + ": Start uploading parameters. "
                "Total loop_num = " + std::to_string(loop_num));
        for (size_t i = 0; i < loop_num; i++) {
            int64_t *key_dst_buf;
            float *value_dst_buf;
            for (size_t k = 0; k < chunk_size; k++) {
                const int64_t key = key_ptr[i * chunk_size + k];
                const size_t global_gpu_id = key % resource_mgr_->get_global_gpu_count();
                const size_t local_gpu_id = resource_mgr_->cal_local_id_from_global_id(global_gpu_id);
                const size_t dst_worker = resource_mgr_->cal_worker_id_from_global_id(global_gpu_id);
                if (dst_worker == worker_id) { // it belongs to this worker
                    key_dst_buf = h_hash_table_key_chunk_per_gpu[local_gpu_id] + 
                                tile_counter_in_chunk_per_gpu[local_gpu_id] * hash_table_key_tile_size;
                    *key_dst_buf = key;

                    value_dst_buf = h_hash_table_value_chunk_per_gpu[local_gpu_id] + 
                                    tile_counter_in_chunk_per_gpu[local_gpu_id] * hash_table_value_tile_size;
                    std::memcpy(value_dst_buf, 
                                embedding_ptr + (i * chunk_size + k) * embedding_vec_size,
                                hash_table_value_tile_size_in_bytes);
                    tile_counter_in_chunk_per_gpu[local_gpu_id]++;
                } else {
                    continue;
                }
            } // for k in chunk_size

            // insert to hash table
            for (size_t id = 0; id < local_gpu_count; id++) {
                const auto& local_gpu = resource_mgr_->get_local_gpu(id);
                context.set_device(local_gpu->get_local_device_id());

                const size_t tile_count = tile_counter_in_chunk_per_gpu[id];
                CK_CUDA(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                        h_hash_table_key_chunk_per_gpu[id],
                                        tile_count * sizeof(int64_t),
                                        cudaMemcpyHostToDevice,
                                        local_gpu->get_stream()));
                
                const size_t value_index_offset = tile_counter_per_gpu[id];
                size_t *value_index_buf = d_hash_table_value_index_chunk_per_gpu[id];
                if (tile_count > 0) {
                    memset_liner(value_index_buf, value_index_offset, 1ul, tile_count,
                                local_gpu->get_stream());
                }

                // do hash table insert <key, index>
                param_->get_hashtable(id)->insert(d_hash_table_key_chunk_per_gpu[id], 
                                                value_index_buf, tile_count,
                                                local_gpu->get_stream());
                param_->get_hashtable(id)->get_and_add_value_head(tile_count, 
                                    local_gpu->get_stream());
            } // for id in local_gpu_count

            // copy embedding vectors
            for (size_t id = 0; id < local_gpu_count; id++) {
                const auto& local_gpu = resource_mgr_->get_local_gpu(id);
                context.set_device(local_gpu->get_local_device_id());

                const size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
                const size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
                const float *src_buf = h_hash_table_value_chunk_per_gpu[id];
                float *dst_buf = param_->get_embedding_table_tensor(id)->GetPtrWithType<float>() + value_chunk_offset;
                CK_CUDA(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                        cudaMemcpyHostToDevice, local_gpu->get_stream()));
            } // for id in local_gpu_count

            resource_mgr_->sync_local_gpus();

            // set counter value
            for (size_t id = 0; id < local_gpu_count; id++) {
                tile_counter_per_gpu[id] += tile_counter_in_chunk_per_gpu[id];
                tile_counter_in_chunk_per_gpu[id] = 0;
                if (tile_counter_per_gpu[id] > param_->get_max_vocabulary_size_per_gpu()) 
                    throw std::runtime_error(ErrorBase + "The size of hash table on GPU " + std::to_string(id) +
                                            " is out of range " + 
                                            std::to_string(param_->get_max_vocabulary_size_per_gpu()));
            } // for id in local_gpu_count
        } // for i in loop_num

        // step 3: process the remaining data (less than a chunk)
        const size_t remain_loop_num = num_total_keys - loop_num * chunk_size;
        int64_t *key_dst_buf;
        size_t *value_index_buf;
        float *value_dst_buf;
        for (size_t i = 0; i < remain_loop_num; i++) {
            const int64_t key = key_ptr[loop_num * chunk_size + i];
            const size_t global_gpu_id = key % resource_mgr_->get_global_gpu_count();
            const size_t local_gpu_id = resource_mgr_->cal_local_id_from_global_id(global_gpu_id);
            const size_t dst_worker = resource_mgr_->cal_worker_id_from_global_id(global_gpu_id);

            if (worker_id == dst_worker) {
                const auto& local_gpu = resource_mgr_->get_local_gpu(local_gpu_id);
                context.set_device(local_gpu->get_local_device_id());

                // copy hashtable key from CPU to GPU
                key_dst_buf = d_hash_table_key_chunk_per_gpu[local_gpu_id];
                CK_CUDA(cudaMemcpyAsync(key_dst_buf, &key, hash_table_key_tile_size_in_bytes,
                                        cudaMemcpyHostToDevice, 
                                        local_gpu->get_stream()));
                
                // set value_index
                const size_t value_index_offset = tile_counter_per_gpu[local_gpu_id];
                value_index_buf = d_hash_table_value_index_chunk_per_gpu[local_gpu_id];
                memset_liner(value_index_buf, value_index_offset, 1ul, 1ul,
                            local_gpu->get_stream());
                
                // hashtable insert
                param_->get_hashtable(local_gpu_id)->insert(d_hash_table_key_chunk_per_gpu[local_gpu_id],
                                                            value_index_buf, hash_table_key_tile_size,
                                                            local_gpu->get_stream());
                param_->get_hashtable(local_gpu_id)->get_and_add_value_head(
                                        hash_table_key_tile_size, local_gpu->get_stream());
                                    
                // memcpy embeddding vectors
                const size_t value_offset = tile_counter_per_gpu[local_gpu_id] * embedding_vec_size;
                value_dst_buf = param_->get_embedding_table_tensor(local_gpu_id)->GetPtrWithType<float>() + value_offset;
                CK_CUDA(cudaMemcpyAsync(value_dst_buf, 
                            embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
                            hash_table_value_tile_size_in_bytes, cudaMemcpyHostToDevice,
                            local_gpu->get_stream()));

                // set counter
                tile_counter_per_gpu[local_gpu_id] += hash_table_key_tile_size;
            } else {
                continue;
            }

            resource_mgr_->sync_local_gpus();
        } // for i in remain_loop_num

        resource_mgr_->sync_all_workers();

        // finally: release temp spaces
        for (size_t id = 0; id < local_gpu_count; id++) {
            context.set_device(resource_mgr_->get_local_gpu(id)->get_local_device_id());

            CK_CUDA(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
            CK_CUDA(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
            CK_CUDA(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
            CK_CUDA(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
        }

    }
private:
    std::shared_ptr<ResourcesManager> resource_mgr_;
    const size_t num_keys_per_rank_;

    // forward spaces
    Tensors2<size_t> mapped_indices_buf_;
    Tensors2<size_t> host_nnz_;
    Tensors2<float> gathered_embeddings_buf_;
    
};

REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DenseGather);

} // namespace SparseOperationKit