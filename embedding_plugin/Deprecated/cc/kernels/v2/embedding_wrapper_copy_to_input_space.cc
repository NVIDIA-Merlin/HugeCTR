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

namespace HugeCTR {
namespace Version2 {

template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::copy_to_input_space(
            const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream,
            const bool input_buffer_reset, int& host_replica_id) {
    /*get input space*/
    const std::string embedding_input_name = is_training ? (embedding_name + "_train") : (embedding_name + "_eval");
    auto input_spaces = get_item_from_map(input_spaces_, embedding_input_name);
    if (!input_spaces) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "Cannot find input_spaces for ", embedding_name);

    /*check whether valid*/
    size_t gpu_count = resource_manager_->get_local_gpu_count();

    auto replica_id_flat = replica_id->flat<int>();
    if (replica_id_flat.size() != 1) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "replica_id should be a scaler.");
    host_replica_id = 0;
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(&host_replica_id, replica_id_flat.data(), 
                sizeof(int) * 1, cudaMemcpyDeviceToHost, tf_stream));
    
    auto nnz_flat = nnz->flat<long long>();
    if (nnz_flat.size() != 1) return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, " ",
                "nnz should be a scaler.");
    long long host_nnz = 0;
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(&host_nnz, nnz_flat.data(), 
                sizeof(long long) * 1, cudaMemcpyDeviceToHost, tf_stream));
    WRAPPER_CUDA_CHECK(cudaStreamSynchronize(tf_stream));
    if (host_replica_id < 0 || host_replica_id >= static_cast<long long>(gpu_count)) return tensorflow::errors::Aborted(
                __FILE__, ":", __LINE__, " replica_id should be in range of [0, gpu_count)");

    HugeCTR::Tensor2<TypeKey> row_offset_buffer = input_spaces->row_offsets_tensors_[host_replica_id];
    HugeCTR::Tensor2<TypeKey> values_buffer = input_spaces->value_tensors_[host_replica_id];
    std::shared_ptr<size_t> nnz_buffer = input_spaces->nnz_array_[host_replica_id];

    auto row_offset_flat = row_offset->flat<TypeKey>();
    auto values_flat = values->flat<TypeKey>();

    if (input_buffer_reset) {
        if (static_cast<size_t>(row_offset_flat.size()) > row_offset_buffer.get_num_elements()) 
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, 
                    " row_offset length outnumbers its embedding buffer.");
        if (static_cast<size_t>(values_flat.size()) > values_buffer.get_num_elements()) 
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, 
                    " values length outnumbers its embedding buffer.");

        /*do copy on plugin stream*/
        auto stream = resource_manager_->get_local_gpu(host_replica_id)->get_stream();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(row_offset_buffer.get_ptr(), row_offset_flat.data(),
                                            row_offset_flat.size() * sizeof(TypeKey),
                                            cudaMemcpyDeviceToDevice, stream));
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(values_buffer.get_ptr(), values_flat.data(),
                                            values_flat.size() * sizeof(TypeKey),
                                            cudaMemcpyDeviceToDevice, stream));
    } else {
        if (static_cast<size_t>(row_offset_flat.size()) < row_offset_buffer.get_num_elements())
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, 
                    " row_offset length is less than its embedding buffer.");
        if (static_cast<size_t>(values_flat.size()) < values_buffer.get_num_elements())
                return tensorflow::errors::Aborted(__FILE__, ":", __LINE__, 
                    " values length is less than its embedding buffer.");

        /*do copy on plugin stream*/
        auto stream = resource_manager_->get_local_gpu(host_replica_id)->get_stream();
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(row_offset_buffer.get_ptr(), row_offset_flat.data(),
                                            row_offset_buffer.get_size_in_bytes(),
                                            cudaMemcpyDeviceToDevice, stream));
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(values_buffer.get_ptr(), values_flat.data(),
                                            values_buffer.get_size_in_bytes(),
                                            cudaMemcpyDeviceToDevice, stream));
    } // if input_buffer_reset

    *nnz_buffer = static_cast<size_t>(host_nnz);

    return tensorflow::Status::OK();
}

template tensorflow::Status EmbeddingWrapper<long long, float>::copy_to_input_space(
            const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream,
            const bool input_buffer_reset, int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<long long, __half>::copy_to_input_space(
            const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream,
            const bool input_buffer_reset, int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::copy_to_input_space(
            const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream,
            const bool input_buffer_reset, int& host_replica_id);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::copy_to_input_space(
            const std::string& embedding_name, 
            const bool is_training, const tensorflow::Tensor* replica_id, 
            const tensorflow::Tensor* row_offset, const tensorflow::Tensor* values, 
            const tensorflow::Tensor* nnz, const cudaStream_t& tf_stream,
            const bool input_buffer_reset, int& host_replica_id);

} // namespace Version2
} // namespace HugeCTR