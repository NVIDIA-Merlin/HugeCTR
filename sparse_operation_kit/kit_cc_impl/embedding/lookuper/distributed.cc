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
#include "embeddings/forward_functions.h"
#include "embeddings/backward_functions.h"

namespace SparseOperationKit {

class DistribtuedLookuper : public EmbeddingLookuper {
public:
    explicit DistribtuedLookuper(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
    : EmbeddingLookuper(context, param), resource_mgr_(context->get_resource_mgr()), 
    max_feature_num_(context->get_max_feature_num()), slot_num_(context->get_slot_num()),
    combiner_(context->get_combiner())
    {
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        hash_value_index_tensors_.reserve(local_gpu_count);
        embedding_feature_tensors_.reserve(local_gpu_count);
        wgrad_tensors_.reserve(local_gpu_count);
        if (combiner_ == CombinerType::Mean) row_offset_allreduce_tensors_.reserve(local_gpu_count);
    }

    void allocate_forward_spaces(size_t const global_batch_size) override {
        size_t max_vocabulary_size_per_gpu = param_->get_max_vocabulary_size_per_gpu();
        size_t max_vocabulary_size_in_total = max_vocabulary_size_per_gpu * resource_mgr_->get_global_gpu_count();

        MESSAGE("max_vocabulary_size_in_total = " + std::to_string(max_vocabulary_size_in_total));

        for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) {
            auto &buffer = base_context()->get_buffer(dev_id);
            // new hash table value (index) that get() from hashtable
            {
                Tensor2<size_t> tensor;
                buffer->reserve({1, global_batch_size * max_feature_num_}, &tensor);
                hash_value_index_tensors_.push_back(tensor);
            #ifdef DEBUG
                std::cout << "hash_value_index_tensor size on dev_id " << dev_id << " = "
                          << "global_batch_size * max_feature_num " << ", "
                          << "global_batch_size = " << global_batch_size << ", "
                          << "max_feature_num = " << max_feature_num_ << std::endl;
            #endif // DEBUG
            }
            // new embedding features reduced by hash table values.
            {
                Tensor2<float> tensor;
                buffer->reserve({global_batch_size * slot_num_, param_->get_embedding_vec_size()}, &tensor);
                embedding_feature_tensors_.push_back(tensor);
            }
        } // for dev_id

        global_batch_size_ = global_batch_size;
    }

    void allocate_backward_spaces(size_t const global_batch_size) override {
        for (size_t dev_id = 0; dev_id < resource_mgr_->get_local_gpu_count(); ++dev_id) { 
            auto &buffer = base_context()->get_buffer(dev_id);
            // new wgrad used by backward
            {
                Tensor2<float> tensor;
                buffer->reserve({global_batch_size * slot_num_, param_->get_embedding_vec_size()}, &tensor);
                wgrad_tensors_.push_back(tensor);
            }
            {
                if (CombinerType::Mean == combiner_) {
                    Tensor2<int64_t> tensor;
                    buffer->reserve({1, global_batch_size * slot_num_ + 1}, &tensor);
                    row_offset_allreduce_tensors_.push_back(tensor);
                } // if combiner_ == mean
            }

        } // for dev_id
    }

    void forward(const Context_t &replica_context, const bool training) override {
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

        const auto &replica_csr_values = replica_context->input("replica_csr_values");
        const auto &replica_row_offset = replica_context->input("replica_row_offset");
        const auto &replica_host_nnz = replica_context->input("replica_host_nnz");

        // get hash_value_index from hash_table by hash_key
        auto &hashtable = param_->get_hashtable(local_replica_id);
        // TODO: if there is no hashtable, directly use valid_values as the hash_value_index_tensor
        if (!hashtable) throw std::runtime_error(ErrorBase + "No available hashtable.");
        if (training) {
            hashtable->get_insert(replica_csr_values->GetPtrWithType<int64_t>(),
                                  hash_value_index_tensors_[local_replica_id].get_ptr(),
                                  replica_host_nnz->GetPtrWithType<size_t>()[0],
                                  local_gpu->get_stream());
        } else {
            hashtable->get(replica_csr_values->GetPtrWithType<int64_t>(),
                        hash_value_index_tensors_[local_replica_id].get_ptr(),
                        replica_host_nnz->GetPtrWithType<size_t>()[0],
                        local_gpu->get_stream());
        }

        replica_context->record_internal_tensor("replica_hash_value_index",
                                                hash_value_index_tensors_[local_replica_id],
                                                /*overwrite=*/true);

        // embedding vector looking up and do reduction
        switch (combiner_) {
            case CombinerType::Sum: {
                forward_sum(/*batch_size=*/global_batch_size_,
                            slot_num_, param_->get_embedding_vec_size(), 
                            /*row_offsets=*/replica_row_offset->GetPtrWithType<int64_t>(),
                            hash_value_index_tensors_[local_replica_id].get_ptr(),
                            /*hash_table_value=*/param_->get_embedding_table_tensor(local_replica_id)->GetPtrWithType<float>(),
                            /*embedding_feature=*/embedding_feature_tensors_[local_replica_id].get_ptr(),
                            local_gpu->get_stream());
                break;
            }
            case CombinerType::Mean: {
                // delay mean scale after reduction-sum
                forward_sum(/*batch_size=*/global_batch_size_,
                            slot_num_, param_->get_embedding_vec_size(),
                            /*row_offsets=*/replica_row_offset->GetPtrWithType<int64_t>(),
                            hash_value_index_tensors_[local_replica_id].get_ptr(),
                            /*hash_table_value=*/param_->get_embedding_table_tensor(local_replica_id)->GetPtrWithType<float>(),
                            /*embedding_feature=*/embedding_feature_tensors_[local_replica_id].get_ptr(),
                            local_gpu->get_stream());
                replica_context->record_internal_tensor("row_offset_allreduce_tensor",
                                row_offset_allreduce_tensors_[local_replica_id]);
                break;
            }
            default: throw std::runtime_error(ErrorBase + "Not supported combiner.");
        } // switch combiner_

        // set outputs
        replica_context->set_output("embedding_features", embedding_feature_tensors_[local_replica_id]);
    }

    void backward(const Context_t &replica_context) override {
        const size_t global_replica_id = replica_context->get_global_replica_id();
        const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
        const auto &stream = resource_mgr_->get_local_gpu(local_replica_id)->get_stream();

        const auto &embedding_features = replica_context->input("embedding_features");

        switch (combiner_) {
            case CombinerType::Sum: {
                backward_sum(/*batch_size=*/global_batch_size_,
                            slot_num_, param_->get_embedding_vec_size(),
                            /*top_grad=*/embedding_features->GetPtrWithType<float>(),
                            /*wgrad=*/wgrad_tensors_[local_replica_id].get_ptr(),
                            stream);
                break;
            }
            case CombinerType::Mean: {
                backward_mean(/*batch_size=*/global_batch_size_,
                            slot_num_, param_->get_embedding_vec_size(),
                            /*row_offset=*/row_offset_allreduce_tensors_[local_replica_id].get_ptr(),
                            /*top_grad=*/embedding_features->GetPtrWithType<float>(),
                            /*wgrad=*/wgrad_tensors_[local_replica_id].get_ptr(),
                            stream);
                break;
            }
            default: throw std::runtime_error(ErrorBase + "Not supported combiner.");
        } // switch combiner_

        // set input grads
        const auto &replica_row_offset = replica_context->input("replica_row_offset");
        auto &replica_input_grad = replica_context->output("replica_input_grad");
        expand_input_grad(global_batch_size_, slot_num_, param_->get_embedding_vec_size(),
                        replica_row_offset->GetPtrWithType<int64_t>(), 
                        wgrad_tensors_[local_replica_id].get_ptr(),
                        replica_input_grad->GetPtrWithType<float>(),
                        stream);

        // set hash_value index
        const auto &replica_hash_value_index = replica_context->get_internal_tensor("replica_hash_value_index");
        auto &value_index_tensor = replica_context->output("value_index_tensor");
        CK_CUDA(cudaMemcpyAsync(value_index_tensor->GetPtrWithType<int64_t>(),
                                replica_hash_value_index->GetPtrWithType<int64_t>(),
                                value_index_tensor->get_size_in_bytes(),
                                cudaMemcpyDeviceToDevice,
                                stream));
    #ifdef DEBUG
        {
            const auto replica_host_nnz = replica_context->input("replica_host_nnz");
            const auto replica_row_offset = replica_context->input("replica_row_offset");
            const auto replica_csr_values = replica_context->input("replica_csr_values");
            const size_t host_nnz = replica_host_nnz->GetPtrWithType<size_t>()[0];
            CK_CUDA(cudaStreamSynchronize(stream));
            
            int64_t *host_row_offset = nullptr;
            CK_CUDA(cudaMallocHost(&host_row_offset, replica_row_offset->get_size_in_bytes(), cudaHostAllocDefault));
            int64_t *host_csr_values = nullptr;
            CK_CUDA(cudaMallocHost(&host_csr_values, sizeof(int64_t) * host_nnz, cudaHostAllocDefault));
            size_t *host_replica_hash_value_index = nullptr;
            CK_CUDA(cudaMallocHost(&host_replica_hash_value_index, sizeof(size_t) * host_nnz, cudaHostAllocDefault));

            CK_CUDA(cudaMemcpyAsync(host_row_offset, 
                                    replica_row_offset->GetPtrWithType<int64_t>(),
                                    replica_row_offset->get_size_in_bytes(),
                                    cudaMemcpyDefault,
                                    stream));
            CK_CUDA(cudaMemcpyAsync(host_csr_values,
                                    replica_csr_values->GetPtrWithType<int64_t>(),
                                    sizeof(int64_t) * host_nnz,
                                    cudaMemcpyDefault,
                                    stream));
            CK_CUDA(cudaMemcpyAsync(host_replica_hash_value_index, 
                                    replica_hash_value_index->GetPtrWithType<size_t>(),
                                    sizeof(size_t) * host_nnz,
                                    cudaMemcpyDefault,
                                    stream));
            CK_CUDA(cudaStreamSynchronize(stream));

            std::cout << "host_row_offset on GPU: " << local_replica_id << std::endl;
            for (size_t i = 0; i < replica_row_offset->get_num_elements(); i++) std::cout << host_row_offset[i] << " ";
            std::cout << std::endl;
            std::cout << "host_csr_values: " << std::endl;
            for (size_t i = 0; i < host_nnz; i++) std::cout << host_csr_values[i] << " ";
            std::cout << std::endl;
            std::cout << "host_hash_value_index: " << std::endl;
            for (size_t i = 0; i < host_nnz; i++) std::cout << host_replica_hash_value_index[i] << " ";
            std::cout << std::endl;
            CK_CUDA(cudaFreeHost(host_row_offset));
            CK_CUDA(cudaFreeHost(host_csr_values));
            CK_CUDA(cudaFreeHost(host_replica_hash_value_index));

            {
                if (local_replica_id == 0) {
                    std::cout << "\nwhole wgrad:" << std::endl;
                    float * host_wgrad = nullptr;
                    CK_CUDA(cudaMallocHost(&host_wgrad, wgrad_tensors_[0].get_size_in_bytes(), cudaHostAllocDefault));
                    CK_CUDA(cudaMemcpyAsync(host_wgrad, wgrad_tensors_[0].get_ptr(),
                                            wgrad_tensors_[0].get_size_in_bytes(),
                                            cudaMemcpyDefault,
                                            stream));
                    CK_CUDA(cudaStreamSynchronize(stream));
                    for (size_t row = 0; row < global_batch_size_ * slot_num_; row++) {
                        std::cout << "Row: " << row << " ";
                        for (size_t col = 0; col < param_->get_embedding_vec_size(); col++) {
                            std::cout << host_wgrad[row * param_->get_embedding_vec_size() + col] << " ";
                        }
                        std::cout << std::endl;
                    }

                    CK_CUDA(cudaFreeHost(host_wgrad));
                }
            }
        }
    #endif // DEBUG
    }

    void load_tensors_to_memory(const std::vector<std::shared_ptr<Tensor>>& tensors) override {
        /*step 1 allocate temp spaces*/
        const size_t embedding_vec_size = param_->get_embedding_vec_size();
        size_t rows_num = 0;
        for (auto tensor : tensors) { 
            size_t row_num = tensor->get_num_elements() / embedding_vec_size;
            rows_num += row_num; 
        } // iter on tensors

        std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>> blobs_buff = 
            HugeCTR::GeneralBuffer2<HugeCTR::CudaHostAllocator>::create();

        Tensor2<int64_t> keys;
        blobs_buff->reserve({rows_num}, &keys);
        Tensor2<float> embedding_vectors;
        blobs_buff->reserve({rows_num, embedding_vec_size}, &embedding_vectors);
        blobs_buff->allocate();
        MESSAGE("Allocated temporary buffer for loading tensors.");

        /*step 2: write content to temporary spaces*/
        size_t row_offset = 0;
        for (auto tensor : tensors) {
            size_t row_num = tensor->get_num_elements() / embedding_vec_size;
            for (size_t i = 0; i < row_num; i++) {
                size_t row_idx = row_offset + i;
                int64_t key = row_idx; // use row-idx as hash_value_index
                std::memcpy(keys.get_ptr() + row_idx, &key, sizeof(int64_t));
                std::memcpy(embedding_vectors.get_ptr() + row_idx * embedding_vec_size,
                            tensor->GetPtrWithType<float>() + i * embedding_vec_size, 
                            sizeof(float) * embedding_vec_size);
            } // for i in row_num
            row_offset += row_num;
        } // iter on tensors
        if (row_offset != rows_num) throw std::runtime_error(ErrorBase + "Error happened in copy tensor content.");

        size_t total_max_vocabulary_size = param_->get_max_vocabulary_size_per_gpu() * resource_mgr_->get_global_gpu_count();
        if (rows_num > total_max_vocabulary_size) throw std::runtime_error(ErrorBase + 
                        "The total rows_num is out of the range of total vocabulary_size of this variable.");
        
        MESSAGE("Total rows_num = " + std::to_string(rows_num) + ", " +
                "while total vocabulary_size = " + std::to_string(total_max_vocabulary_size));

        load_parameters(keys, embedding_vectors, rows_num, 
                        total_max_vocabulary_size,
                        param_->get_embedding_vec_size(),
                        param_->get_max_vocabulary_size_per_gpu());
    }

private:
    std::shared_ptr<ResourcesManager> resource_mgr_;
    const size_t max_feature_num_;
    const size_t slot_num_;
    CombinerType combiner_;
    size_t global_batch_size_ = 0;

    // forward spaces
    Tensors2<size_t> hash_value_index_tensors_;
    Tensors2<float> embedding_feature_tensors_;

    // backward spaces
    Tensors2<float> wgrad_tensors_;
    Tensors2<int64_t> row_offset_allreduce_tensors_;

    void load_parameters(const Tensor2<int64_t>& keys,
                        const Tensor2<float>& embeddings,
                        size_t row_num,
                        size_t vocabulary_size,
                        size_t embedding_vec_size,
                        size_t max_vocabulary_size_per_gpu) {
        if (keys.get_dimensions()[0] != row_num || embeddings.get_dimensions()[0] != row_num) 
            throw std::runtime_error(ErrorBase + "The rows of keys and embeddings are not consistent.");
        if (row_num > vocabulary_size)
            throw std::runtime_error(ErrorBase + "file size is larger than vocabulary_size.");

        const int64_t *key_ptr = keys.get_ptr();
        const float *embedding_ptr = embeddings.get_ptr();

        /*step 5: allocate temporary spaces*/
        const size_t worker_id = resource_mgr_->get_worker_id();
        const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
        const size_t chunk_size = 1000;
        size_t hash_table_key_tile_size = 1; // ????
        size_t hash_table_key_tile_size_in_bytes = hash_table_key_tile_size * sizeof(int64_t); // TODO: make it template
        size_t hash_table_key_chunk_size = hash_table_key_tile_size * chunk_size;
        size_t hash_table_key_chunk_size_in_bytes = hash_table_key_chunk_size * sizeof(int64_t);
        size_t hash_table_value_index_chunk_size_in_bytes = hash_table_key_chunk_size * sizeof(size_t);
        size_t hash_table_value_tile_size = embedding_vec_size;
        size_t hash_table_value_tile_size_in_bytes = hash_table_value_tile_size * sizeof(float);
        size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
        size_t hash_table_value_chunk_size_in_bytes = hash_table_value_chunk_size * sizeof(float);

        // Cannot decide precise the number of values for each GPU, so allocate enough spaces
        std::unique_ptr<size_t []> tile_counter_per_gpu(new size_t[local_gpu_count]()); // hash_table_value_index_per_gpu_size
        std::memset(tile_counter_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
        std::unique_ptr<size_t []> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]());
        std::memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);
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

        /*step 6: do uploading*/
        size_t loop_num = row_num / chunk_size;
        MESSAGE("Rank " + std::to_string(worker_id) + ": Start uploading parameters. "
                "Total loop_num = " + std::to_string(loop_num));
        for (size_t i = 0; i < loop_num; i++) {
            int64_t *key_dst_buf;
            float *value_dst_buf;
            for (size_t k = 0; k < chunk_size; k++) {
                int64_t key = key_ptr[i * chunk_size + k];
                size_t global_gpu_id = key % resource_mgr_->get_global_gpu_count();
                size_t local_gpu_id = resource_mgr_->cal_local_id_from_global_id(global_gpu_id);
                size_t dst_worker = resource_mgr_->cal_worker_id_from_global_id(global_gpu_id);
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

                size_t tile_count = tile_counter_in_chunk_per_gpu[id];
                CK_CUDA(cudaMemcpyAsync(d_hash_table_key_chunk_per_gpu[id],
                                        h_hash_table_key_chunk_per_gpu[id],
                                        tile_count * sizeof(int64_t),
                                        cudaMemcpyHostToDevice,
                                        local_gpu->get_stream()));
                
                size_t value_index_offset = tile_counter_per_gpu[id];
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

                size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * embedding_vec_size;
                size_t value_chunk_offset = tile_counter_per_gpu[id] * embedding_vec_size;
                float *src_buf = h_hash_table_value_chunk_per_gpu[id];
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

        /*step 7: process the remaining data (less than a chunk)*/
        size_t remain_loop_num = row_num - loop_num * chunk_size;
        int64_t *key_dst_buf;
        size_t *value_index_buf;
        float *value_dst_buf;
        for (size_t i = 0; i < remain_loop_num; i++) {
            int64_t key = key_ptr[loop_num * chunk_size + i];
            size_t global_gpu_id = key % resource_mgr_->get_global_gpu_count();
            size_t local_gpu_id = resource_mgr_->cal_local_id_from_global_id(global_gpu_id);
            size_t dst_worker = resource_mgr_->cal_worker_id_from_global_id(global_gpu_id);

            if (worker_id == dst_worker) {
                const auto& local_gpu = resource_mgr_->get_local_gpu(local_gpu_id);
                context.set_device(local_gpu->get_local_device_id());

                // copy hashtable key from CPU to GPU
                key_dst_buf = d_hash_table_key_chunk_per_gpu[local_gpu_id];
                CK_CUDA(cudaMemcpyAsync(key_dst_buf, &key, hash_table_key_tile_size_in_bytes,
                                        cudaMemcpyHostToDevice, 
                                        local_gpu->get_stream()));
                
                // set value_index
                size_t value_index_offset = tile_counter_per_gpu[local_gpu_id];
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
                size_t value_offset = tile_counter_per_gpu[local_gpu_id] * embedding_vec_size;
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


        /*step 8: use nccl collective operation to synchronize all machines*/
        CK_NCCL(ncclGroupStart());
        for (size_t id = 0; id < local_gpu_count; id++) {
            const auto& local_gpu = resource_mgr_->get_local_gpu(id);
            CK_NCCL(ncclBroadcast(d_hash_table_value_index_chunk_per_gpu[id], 
                                d_hash_table_value_index_chunk_per_gpu[id], 
                                1, ncclUint64, /*root=*/0,
                                local_gpu->get_nccl(),
                                local_gpu->get_stream()));
        }
        CK_NCCL(ncclGroupEnd());
        resource_mgr_->sync_local_gpus();


        /*finally: release temp spaces*/
        for (size_t id = 0; id < local_gpu_count; id++) {
            context.set_device(resource_mgr_->get_local_gpu(id)->get_local_device_id());

            CK_CUDA(cudaFree(d_hash_table_value_index_chunk_per_gpu[id]));
            CK_CUDA(cudaFreeHost(h_hash_table_key_chunk_per_gpu[id]));
            CK_CUDA(cudaFree(d_hash_table_key_chunk_per_gpu[id]));
            CK_CUDA(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
        }

    }

};

REGISTER_EMB_LOOKUPER_BUILDER("distributed", DistribtuedLookuper);

} // namespace SparseOperationKit