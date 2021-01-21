/*
* Copyright (c) 2020, NVIDIA CORPORATION.
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


#include "embedding_wrapper.h"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"
#include "embedding_utils.hpp"

namespace HugeCTR {
namespace Version1 {


/** This function is used to distribute keys to each GPU as CSR format.
* This function is similar to data_reader_worker.hpp:read_a_batch()
* @param sparse_indices, @param values, @param dense_shape, these three tensors made of tf.sparseTensor, and the shape of corresponding
* dense tensor is [N, slot_num, max_nnz].
* @param sparse_indices, [N, ndims], N = valid key_num and ndims = 3, which represents dimension: {batch, slot_num, max_nnz},
* dim0 could be used to decide sample index, dim1 could be used to decide slot index.
* @param values, [N] = number of keys
* @param dense_shape, [ndims], each dimension in this dense_shape
* @param embedding_name, used to find the input space for this embedding instance.
*/ 
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys(const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu) {
    std::shared_ptr<EmbeddingParams> params = get_embedding_params(embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, " Not found embedding params for ", embedding_name);

    /*get input space*/
    std::string input_space_name = embedding_name;
    input_space_name += (is_training ? "_train" : "_eval");
    std::shared_ptr<InputSpace> space = get_input_space(input_space_name);
    if (!space) return tensorflow::errors::NotFound(__FILE__, ": ", __LINE__, ", Did not find ", 
                                                    input_space_name, " in input_spaces.");

    /*check dims*/
    if (sparse_indices->dims() != 2) return tensorflow::errors::Unavailable("sparse_indices.dims should be 2, but got ", 
                                sparse_indices->dims());
    if (values->dims() != 1) return tensorflow::errors::Unavailable("values.dims should be 1, but got ", values->dims());
    if (dense_shape->dims() != 1) return tensorflow::errors::Unavailable("dense_shape.dims should be 1, but got ", dense_shape->dims());
    if (dense_shape->dim_size(0) != 3) return tensorflow::errors::Unavailable("dense_shape.num_elements should be 3. but got ",
                                                        dense_shape->dim_size(0));
    long long samples_num = 0;
    long long slot_num = 0;
    long long sample_max_nnz = 0;
    if (on_gpu) {
        std::unique_ptr<long long []> host_dense_shape_tensor(new long long[dense_shape->dim_size(0)]());
        WRAPPER_CUDA_CHECK(cudaMemcpy(host_dense_shape_tensor.get(), dense_shape->flat<long long>().data(),
                                       sizeof(long long) * dense_shape->dim_size(0),
                                       cudaMemcpyDeviceToHost));
        samples_num = host_dense_shape_tensor[0];
        slot_num = host_dense_shape_tensor[1];
        sample_max_nnz = host_dense_shape_tensor[2];
    } else { // on cpu
        auto dense_shape_tensor = dense_shape->tensor<long long, 1>();
        samples_num = dense_shape_tensor(0);
        slot_num = dense_shape_tensor(1);
        sample_max_nnz = dense_shape_tensor(2);
    }
    if (samples_num != (is_training ? batch_size_ : batch_size_eval_))
        return tensorflow::errors::OutOfRange(__FILE__, ": ", __LINE__, " This input's batch_size ",
                                                    " is not equal to which is initialized.");
    if (samples_num % resource_manager_->get_global_gpu_count() != 0)
        return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ", "input's batch_size must be a multiple of GPU counts, "
                                            "otherwise batch_size / gpu_count may be 0.");
    if (slot_num != params->slot_num_) return tensorflow::errors::Unavailable(__FILE__, ": ", __LINE__, 
                                        " input keys's slot_num is not equal to which is initialized.");
    if (sample_max_nnz > (params->max_feature_num_ > (params->max_nnz_ * params->slot_num_) ? 
            params->max_feature_num_ : (params->max_nnz_ * params->slot_num_)))
        return tensorflow::errors::OutOfRange(__FILE__, ": ", __LINE__, " input_keys's nnz:", sample_max_nnz,
        " is large than max_feature_num:", params->max_feature_num_, " or max_nnz:", params->max_nnz_, ".");

    /*create temp csr_chunk*/
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    std::vector<std::unique_ptr<TypeKey []>> row_offsets_(gpu_count);
    std::vector<std::unique_ptr<TypeKey []>> value_tensors_(gpu_count);
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        row_offsets_[dev_id].reset(new TypeKey[space->row_offsets_tensors_[dev_id].get_num_elements()]());
        value_tensors_[dev_id].reset(new TypeKey[space->value_tensors_[dev_id].get_num_elements()]());
    }
    /*apply to csr_chunk reset*/
    std::vector<int> size_of_row_offset_(gpu_count, 0);
    std::vector<int> size_of_value_(gpu_count, 0);

    /*decide each key's GPU location.*/
    long long* indices_ptr = nullptr;
    long long* keys_ptr = nullptr;
    auto sparse_indices_flat = sparse_indices->flat<long long>();
    auto values_flat = values->flat<long long>();
    std::unique_ptr<long long []> host_indices;
    std::unique_ptr<long long []> host_keys; 
    if (on_gpu) {
        host_indices.reset(new long long[sparse_indices_flat.size()]());
        WRAPPER_CUDA_CHECK(cudaMemcpy(host_indices.get(), sparse_indices_flat.data(), 
                                       sizeof(long long) * sparse_indices_flat.size(),
                                       cudaMemcpyDeviceToHost));
        
        host_keys.reset(new long long[values_flat.size()]());
        WRAPPER_CUDA_CHECK(cudaMemcpy(host_keys.get(), values_flat.data(),
                           sizeof(long long) * values_flat.size(),
                           cudaMemcpyDeviceToHost));

        indices_ptr = host_indices.get();
        keys_ptr = host_keys.get();
    } else { // on cpu
        indices_ptr = const_cast<long long*>(sparse_indices_flat.data());
        keys_ptr = const_cast<long long*>(values_flat.data());
    }
    long long key_idx = 0;
    for (long long sample_id = 0; sample_id < samples_num; sample_id++) { // batch loop
        for (long long slot_id = 0; slot_id < slot_num; slot_id++) { // slot loop
            switch (embedding_type) {
                case HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash: {
                    // new row
                    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
                        row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<TypeKey>(size_of_value_[dev_id]);
                    }

                    // keys belong to same sample and same slot, which means calculate nnz in this slot
                    while (true) {
                        if (key_idx < values->dim_size(0) && 
                            sample_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 0] && 
                            slot_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 1]) {
                            unsigned int dev_id = keys_ptr[key_idx] % gpu_count;
                            value_tensors_[dev_id][size_of_value_[dev_id]++] = 
                                    static_cast<TypeKey>(keys_ptr[key_idx]); // push_back
                            ++key_idx;
                        } else { // keys belong to different sample or slot
                            break;
                        }
                    } // while
                    break;
                }
                case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot:
                case HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash: {
                    // new row
                    unsigned int dev_id = slot_id % gpu_count;
                    row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<TypeKey>(size_of_value_[dev_id]);

                    // keys belong to same sample and same slot, which means calculate nnz in this slot
                    while (true) {
                         if (key_idx < values->dim_size(0) && 
                            sample_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 0] && 
                            slot_id == indices_ptr[key_idx * dense_shape->dim_size(0) + 1]) {
                            value_tensors_[dev_id][size_of_value_[dev_id]++] = 
                                    static_cast<TypeKey>(keys_ptr[key_idx]);
                            ++key_idx;
                        } else { // keys belong to different sample or slot
                            break;
                        }
                    } // while
                    break;
                }
                default: {
                    return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                       "Unsupported embedding type.");
                    break;
                }
            } // switch
        } // slot loop
    } // batch loop

    /*write the last index to row*/
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        row_offsets_[dev_id][size_of_row_offset_[dev_id]++] = static_cast<TypeKey>(size_of_value_[dev_id]);
    }
    
    /*copy to each GPU's input space*/
    HugeCTR::CudaDeviceContext context;
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        context.set_device(cur_device);

        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(space->row_offsets_tensors_[dev_id].get_ptr(), row_offsets_[dev_id].get(),
                        space->row_offsets_tensors_[dev_id].get_size_in_bytes(), cudaMemcpyHostToDevice, 
                        resource_manager_->get_local_gpu(dev_id)->get_stream()));

        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(space->value_tensors_[dev_id].get_ptr(), value_tensors_[dev_id].get(),
                        space->value_tensors_[dev_id].get_size_in_bytes(), cudaMemcpyHostToDevice,
                        resource_manager_->get_local_gpu(dev_id)->get_stream()));

        /*write nnz buffer*/
        *(space->nnz_array_[dev_id]) = size_of_value_[dev_id];
    }

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        int cur_device = resource_manager_->get_local_gpu(dev_id)->get_device_id();
        context.set_device(cur_device);
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(resource_manager_->get_local_gpu(dev_id)->get_stream()));
    }

#ifndef NDEBUG
    /*temp print*/
    std::cout << "row_offset & value for this input sample..." << std::endl;
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        std::cout << "GPU: " << dev_id << std::endl;
        std::cout << "row_offsets_ = ";
        for (size_t i = 0; i < space->row_offsets_tensors_[dev_id].get_num_elements(); ++i) {
            std::cout << row_offsets_[dev_id][i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "value_ = ";
        for (size_t i = 0; i < space->value_tensors_[dev_id].get_num_elements(); ++i){
            std::cout << value_tensors_[dev_id][i] << ", ";
        }
        std::cout << std::endl;
    }
#endif

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::distribute_keys(
            const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<long long, __half>::distribute_keys(
            const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::distribute_keys(
            const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::distribute_keys(
            const tensorflow::Tensor* sparse_indices, 
            const tensorflow::Tensor* values, const tensorflow::Tensor* dense_shape, 
            const std::string& embedding_name, const bool is_training,
            const HugeCTR::Embedding_t& embedding_type, const bool on_gpu);


} // namespace Version1
} // namespace HugeCTR