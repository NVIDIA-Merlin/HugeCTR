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

/* TODO: IMPORTANT:
* In cusparseXcoo2csr, cooRowInd is int*, 
* therefore its range is limited.
*/
template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys_gpu_distributed(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space) {
    /*get internel spaces & reset*/                                                    
    auto internel_spaces = get_item_from_map(emb_distribute_keys_internel_spaces_,
                                            embedding_name);
    if (!internel_spaces) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                    "Do not find this internel_spaces");
    WRAPPER_REQUIRE_OK(internel_spaces->reset());

    /*get embedding hyper params*/
    auto params = get_item_from_map(embedding_params_, embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                    "Do not find hyper parameters for this embedding.");

    /*copy input to internel spaces*/
    HugeCTR::CudaDeviceContext context;
    // CPU -> GPU0
    const auto& local_gpu = resource_manager_->get_local_gpu(0); 
    context.set_device(local_gpu->get_device_id());
    auto row_indices_flat = row_indices->flat<long long>();
    auto value_flat = values->flat<TypeKey>();
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->copy_input_row_indices_[0].get_ptr(),
                                       row_indices_flat.data(),
                                       row_indices_flat.size() * sizeof(long long),
                                       cudaMemcpyDeviceToDevice, 
                                       local_gpu->get_stream()));
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->copy_input_values_[0].get_ptr(),
                                       value_flat.data(),
                                       value_flat.size() * sizeof(TypeKey),
                                       cudaMemcpyDeviceToDevice,
                                       local_gpu->get_stream()));
    // BroadCast to all GPU
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    if (gpu_count > 1) {
        WRAPPER_NCCL_CHECK(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
            const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
            WRAPPER_NCCL_CHECK(ncclBcast(internel_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                         row_indices_flat.size(),
                                         ncclInt64, 0/*root*/, 
                                         local_gpu->get_nccl(), local_gpu->get_stream()));
            WRAPPER_NCCL_CHECK(ncclBcast(internel_spaces->copy_input_values_[dev_id].get_ptr(),
                                         value_flat.size(),
                                         nccl_type_, 0/*root*/,
                                         local_gpu->get_nccl(), local_gpu->get_stream()));    
        } // for dev_id
        WRAPPER_NCCL_CHECK(ncclGroupEnd());
    } // if gpu_count > 1


    /*Iter each GPU to launch kernel.*/
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*generate binary flag vector based on values*/
        CudaUtils::distributed_binary_vector(internel_spaces->copy_input_values_[dev_id].get_ptr(),
                                            value_flat.size(), 
                                            gpu_count, dev_id,
                                            internel_spaces->binary_flags_[dev_id].get_ptr(),
                                            local_gpu->get_stream());

        /*choose values*/
        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(internel_spaces->cub_d_temp_storage_[dev_id],
                                                  internel_spaces->cub_temp_storage_bytes_[dev_id],
                                                  internel_spaces->copy_input_values_[dev_id].get_ptr(),
                                                  internel_spaces->binary_flags_[dev_id].get_ptr(),
                                                  internel_spaces->cub_values_output_[dev_id].get_ptr(),
                                                  internel_spaces->cub_dev_num_selected_[dev_id],
                                                  value_flat.size(),
                                                  local_gpu->get_stream()));

        /*choose row_indices*/
        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(internel_spaces->cub_d_temp_storage_[dev_id],
                                                  internel_spaces->cub_temp_storage_bytes_[dev_id],
                                                  internel_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                                  internel_spaces->binary_flags_[dev_id].get_ptr(),
                                                  internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                                  internel_spaces->cub_dev_num_selected_[dev_id],
                                                  row_indices_flat.size(),
                                                  local_gpu->get_stream()));

        /*copy num_selected from dev to CPU*/
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->cub_host_num_selected_[dev_id],
                                           internel_spaces->cub_dev_num_selected_[dev_id],
                                           sizeof(size_t) * 1,
                                           cudaMemcpyDeviceToHost,
                                           local_gpu->get_stream()));
    } // for dev_id

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*need stream synchronize here to make sure cub_host_num_selected_ is correctly copy back to CPU*/
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));
    } // for dev_id

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*convert COO row_indices to CSR row_offsets*/
        int rows_num = ((is_training ? batch_size_ : batch_size_eval_) * params->slot_num_);
        WRAPPER_CUSPARSE_CHECK(cusparseXcoo2csr(internel_spaces->cusparse_handles_[dev_id],
                                                internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                                *(internel_spaces->cub_host_num_selected_[dev_id]),
                                                rows_num,
                                                internel_spaces->cusparse_csr_row_offsets_output_[dev_id].get_ptr(),
                                                CUSPARSE_INDEX_BASE_ZERO));

        /*cast row_offsets*/
        CudaUtils::cast_elements(internel_spaces->cusparse_csr_row_offsets_output_[dev_id].get_ptr(),
                                 internel_spaces->csr_row_offsets_cast_[dev_id].get_ptr(),
                                 rows_num + 1,
                                 32/*sm count*/, local_gpu->get_stream());

        /*copy row_offsets and values to embedding input space*/
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(input_space->row_offsets_tensors_[dev_id].get_ptr(),
                                           internel_spaces->csr_row_offsets_cast_[dev_id].get_ptr(),
                                           input_space->row_offsets_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(input_space->value_tensors_[dev_id].get_ptr(),
                                           internel_spaces->cub_values_output_[dev_id].get_ptr(),
                                           input_space->value_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));
        /*write nnz to input space*/
        *(input_space->nnz_array_[dev_id]) = *(internel_spaces->cub_host_num_selected_[dev_id]);
    } // for dev_id
    
    return tensorflow::Status::OK();
}


template <typename TypeKey, typename TypeFP>
tensorflow::Status EmbeddingWrapper<TypeKey, TypeFP>::distribute_keys_gpu_localized(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space) {
    /*get internel spaces & reset*/                                                    
    auto internel_spaces = get_item_from_map(emb_distribute_keys_internel_spaces_,
                                            embedding_name);
    if (!internel_spaces) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                    "Do not find this internel_spaces");
    WRAPPER_REQUIRE_OK(internel_spaces->reset());

    /*get embedding hyper params*/
    auto params = get_item_from_map(embedding_params_, embedding_name);
    if (!params) return tensorflow::errors::NotFound(__FILE__, ":", __LINE__, " ",
                                    "Do not find hyper parameters for this embedding.");

    /*copy input to internel spaces*/
    HugeCTR::CudaDeviceContext context;
    // CPU -> GPU0
    const auto& local_gpu = resource_manager_->get_local_gpu(0); 
    context.set_device(local_gpu->get_device_id());
    auto row_indices_flat = row_indices->flat<long long>();
    auto value_flat = values->flat<TypeKey>();

    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->copy_input_values_[0].get_ptr(),
                                       value_flat.data(),
                                       value_flat.size() * sizeof(TypeKey),
                                       cudaMemcpyDeviceToDevice,
                                       local_gpu->get_stream()));
    WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->copy_input_row_indices_[0].get_ptr(),
                                       row_indices_flat.data(),
                                       row_indices_flat.size() * sizeof(long long),
                                       cudaMemcpyDefault, 
                                       local_gpu->get_stream()));

    // BroadCast to all GPU
    size_t gpu_count = resource_manager_->get_local_gpu_count();
    if (gpu_count > 1) {
        WRAPPER_NCCL_CHECK(ncclGroupStart());
        for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
            const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
            WRAPPER_NCCL_CHECK(ncclBcast(internel_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                         row_indices_flat.size(),
                                         ncclInt64, 0/*root*/, 
                                         local_gpu->get_nccl(), local_gpu->get_stream()));
            WRAPPER_NCCL_CHECK(ncclBcast(internel_spaces->copy_input_values_[dev_id].get_ptr(),
                                         value_flat.size(),
                                         nccl_type_, 0/*root*/,
                                         local_gpu->get_nccl(), local_gpu->get_stream()));             
        } // for dev_id
        WRAPPER_NCCL_CHECK(ncclGroupEnd());
    } // if gpu_count > 1

    /*Iter each GPU to launch kernel*/
    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*generate binary flag based on COO row_indices*/
        CudaUtils::localized_binary_vector(internel_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                           row_indices_flat.size(),
                                           gpu_count, dev_id, params->slot_num_,
                                           internel_spaces->binary_flags_[dev_id].get_ptr(),
                                           local_gpu->get_stream());

        /*choose row_indices*/
        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(internel_spaces->cub_d_temp_storage_[dev_id],
                                                  internel_spaces->cub_temp_storage_bytes_[dev_id],
                                                  internel_spaces->copy_input_row_indices_[dev_id].get_ptr(),
                                                  internel_spaces->binary_flags_[dev_id].get_ptr(),
                                                  internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                                  internel_spaces->cub_dev_num_selected_[dev_id],
                                                  row_indices_flat.size(),
                                                  local_gpu->get_stream()));
        
        /*choose values*/
        WRAPPER_CUDA_CHECK(CudaUtils::cub_flagged(internel_spaces->cub_d_temp_storage_[dev_id],
                                                  internel_spaces->cub_temp_storage_bytes_[dev_id],
                                                  internel_spaces->copy_input_values_[dev_id].get_ptr(),
                                                  internel_spaces->binary_flags_[dev_id].get_ptr(),
                                                  internel_spaces->cub_values_output_[dev_id].get_ptr(),
                                                  internel_spaces->cub_dev_num_selected_[dev_id],
                                                  value_flat.size(),
                                                  local_gpu->get_stream()));

        /*copy num_selected from dev to CPU*/
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(internel_spaces->cub_host_num_selected_[dev_id],
                                           internel_spaces->cub_dev_num_selected_[dev_id],
                                           sizeof(size_t) * 1,
                                           cudaMemcpyDeviceToHost,
                                           local_gpu->get_stream()));

    } // for dev_id

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*need stream synchronize here to make sure cub_host_num_selected_ is correctly copy back to CPU*/
        WRAPPER_CUDA_CHECK(cudaStreamSynchronize(local_gpu->get_stream()));
    } // for dev_id

    for (size_t dev_id = 0; dev_id < gpu_count; ++dev_id) {
        const auto& local_gpu = resource_manager_->get_local_gpu(dev_id);
        context.set_device(local_gpu->get_device_id());

        /*recompute row_indices*/
        CudaUtils::localized_new_row_indices(internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                             internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                             params->slot_num_, internel_spaces->dev_slot_num_[dev_id],
                                             gpu_count, 
                                             *(internel_spaces->cub_host_num_selected_[dev_id]),
                                             local_gpu->get_stream());


        /*convert COO row_indices to CSR row_offsets*/
        int rows_num = ((is_training ? batch_size_ : batch_size_eval_) * internel_spaces->dev_slot_num_[dev_id]);
        WRAPPER_CUSPARSE_CHECK(cusparseXcoo2csr(internel_spaces->cusparse_handles_[dev_id],
                                                internel_spaces->cub_coo_indices_output_[dev_id].get_ptr(),
                                                *(internel_spaces->cub_host_num_selected_[dev_id]),
                                                rows_num,
                                                internel_spaces->cusparse_csr_row_offsets_output_[dev_id].get_ptr(),
                                                CUSPARSE_INDEX_BASE_ZERO));

        /*cast row_offsets*/
        CudaUtils::cast_elements(internel_spaces->cusparse_csr_row_offsets_output_[dev_id].get_ptr(),
                                 internel_spaces->csr_row_offsets_cast_[dev_id].get_ptr(),
                                 rows_num + 1,
                                 32/*sm count*/, local_gpu->get_stream());

        /*copy row_offsets and values to embedding input space*/
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(input_space->row_offsets_tensors_[dev_id].get_ptr(),
                                           internel_spaces->csr_row_offsets_cast_[dev_id].get_ptr(),
                                           input_space->row_offsets_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));
        WRAPPER_CUDA_CHECK(cudaMemcpyAsync(input_space->value_tensors_[dev_id].get_ptr(),
                                           internel_spaces->cub_values_output_[dev_id].get_ptr(),
                                           input_space->value_tensors_[dev_id].get_size_in_bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           local_gpu->get_stream()));
        /*write nnz to input space*/
        *(input_space->nnz_array_[dev_id]) = *(internel_spaces->cub_host_num_selected_[dev_id]);

    } // for dev_id

    return tensorflow::Status::OK();
}


template tensorflow::Status EmbeddingWrapper<long long, float>::distribute_keys_gpu_distributed(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<long long, __half>::distribute_keys_gpu_distributed(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::distribute_keys_gpu_distributed(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::distribute_keys_gpu_distributed(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);

template tensorflow::Status EmbeddingWrapper<long long, float>::distribute_keys_gpu_localized(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<long long, __half>::distribute_keys_gpu_localized(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<unsigned int, float>::distribute_keys_gpu_localized(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);
template tensorflow::Status EmbeddingWrapper<unsigned int, __half>::distribute_keys_gpu_localized(
                                                    const tensorflow::Tensor* row_indices,
                                                    const tensorflow::Tensor* values,
                                                    const std::string& embedding_name,
                                                    const bool is_training,
                                                    std::shared_ptr<InputSpace>& input_space);

} // namespace Version1
} // namespace HugeCTR